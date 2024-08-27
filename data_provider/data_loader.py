import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.calibration import LabelEncoder
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings

warnings.filterwarnings('ignore')

import os
from os.path import join
from utils.data_normalization import adaptive_normal
from monai.transforms import (
    Compose,
    LoadImaged,
    ToTensord,
    EnsureChannelFirstd,
    Resized,
)


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    # def __init__(self, root_path, file_list=None, limit_size=None, flag=None):
    def __init__(self, root_path, table_path, image_path, file_list=None, limit_size=None, flag=None):
        # print("root_path: ", root_path)
        # print("table_path: ", table_path)
        # print("image_path: ", image_path)
        self.root_path = root_path
        self.all_df, self.labels_df, self.image_table_paths = self.load_all(root_path, file_list=file_list, flag=flag)
        # print(self.image_table_paths)

        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        # print("all_IDs: ", len(self.all_IDs))

        #============
        # image_embedding
        self.image_input_dim = (1, 256, 32, 32)
        self.hidden_dim = 256
        # self.image_input = torch.randn(3, *self.image_input_dim)
        # print(self.image_input.shape)

        # self.mri_nii = glob.glob(join(image_path, '*.nii.gz'))
        image_path=os.path.abspath(image_path)
        # print(self.image_table_paths)
        self.mri_nii = [glob.glob(os.path.join(image_path, f"{name.split('/')[-1]}.nii.gz")) for name in self.image_table_paths]
        self.start_transformer = LoadImaged(keys=['image'])
        desired_shape = (256, 32, 32)
        self.transformer = Compose(
            [
                EnsureChannelFirstd(keys=['image']),
                Resized(keys=['image'], spatial_size=desired_shape),
                ToTensord(keys=['image'])
            ])

        #============
        # table_embedding
        # read excel here
        excel_file = os.path.abspath(table_path)
        # drop column: number, label
        # df = pd.read_excel(excel_file, usecols=lambda x: (x != 0 and x != 'number' and x != 'label'))
        df= pd.read_excel(excel_file)
        # Separate the number column for later use and drop label
        self.number_to_index = {row['NUMBER']: idx for idx, row in df.iterrows()}
        df = df.drop(columns=['NUMBER', 'NAME'])

        numeric_columns = []
        categorical_columns = []
        column_types = df.dtypes
        for column_name, dtype in column_types.items():
            if dtype == 'object':
                categorical_columns.append(column_name)
            else:
                numeric_columns.append(column_name)
        # end read

        # Norm numeric data
        df_numeric = df[numeric_columns]
        means = df_numeric.mean()
        stds = df_numeric.std()
        df_numeric_normalized = (df_numeric - means) / (stds + 1e-8)
        # print("df_numeric_normalized: ", df_numeric_normalized.shape)

        # one-hot encode for categorical data
        df_categorical = df[categorical_columns]
        # df_categorical_encoded = pd.get_dummies(df_categorical)
        # print("df_categorical_encoded: ", df_categorical_encoded.shape)
        # 替代one-hot编码
        self.num_cat = []
        for col in df_categorical.columns:
            df_categorical[col] = LabelEncoder().fit_transform(df_categorical[col])
            self.num_cat.append(len(df_categorical[col].unique()))
        df_categorical_encoded = df_categorical
        # print("df_categorical_encoded: ", df_categorical_encoded.shape)

        # to tensor
        self.numeric_input = torch.tensor(df_numeric_normalized.values, dtype=torch.float32)
        self.categorical_input = torch.tensor(df_categorical_encoded.values, dtype=torch.long)
        # print(self.numeric_input.shape)
        # print(self.categorical_input.shape)

        self.numeric_input_dim = len(df_numeric_normalized.columns)
        self.categorical_input_dim = len(df_categorical_encoded.columns)
        self.num_cont = self.numeric_input_dim

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from csv files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .csv files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """

        folders = os.listdir(root_path)
        class_names = [folder for folder in folders if os.path.isdir(os.path.join(root_path, folder))]

        data_paths = []  # list of item path
        labels = []  # corrsponding class name i.e. folder name

        # image and table
        image_table_paths = []

        for class_name in class_names:
            classed_data_paths = glob.glob(os.path.join(root_path, class_name, '*'))

            classed_data_paths = [item for item in classed_data_paths if os.path.isfile(item) and item.endswith('.csv')]
            data_paths += classed_data_paths
            labels += [class_name] * len(classed_data_paths)

            # image and table
            tem_paths = [filename.replace(".csv", "") for filename in classed_data_paths]
            tem_paths = [filename.replace("./datasets/EEG/train/AD\\", "") for filename in tem_paths]
            tem_paths = [filename.replace("./datasets/EEG/train/MCI\\", "") for filename in tem_paths]
            tem_paths = [filename.replace("./datasets/EEG/train/NC\\", "") for filename in tem_paths]
            image_table_paths += tem_paths

        if len(data_paths) == 0:
            raise Exception('No file found in root_path')

        labels = pd.Series(labels, dtype='category')
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int8)

        df_list = [pd.read_csv(p) for p in data_paths]

        df_lengths = [len(x) for x in df_list]
        self.max_seq_len = max(df_lengths)

        horiz_lenghts = [len(x.columns) for x in df_list]
        if len(set(horiz_lenghts)) != 1:
            df_list = [df.apply(subsample) for df in df_list]

        df_index = [i for i, x in enumerate(df_lengths) for _ in range(x)]

        # Generate new df_lengths and df_index after subsample
        df_lengths = [len(x) for x in df_list]
        df_index = [i for i, x in enumerate(df_lengths) for _ in range(x)]

        df = pd.concat(df_list).reset_index(drop=True).set_index(
            pd.Series(df_index)
        )

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df, image_table_paths

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        # print(ind)
        mri_path = self.mri_nii[ind]
        batch = self.start_transformer(dict(image=mri_path))
        batch['image'] = adaptive_normal(batch['image'])
        batch = self.transformer(batch)
        batch['image'] = batch["image"][:1, ...]
        # batch['label'] = int(re.findall('-(\d).nii.gz', mri_path)[0])
        # batch['name'] = mri_path.split('/')[-1]

        # Extract the number from the image filename
        number = int(re.findall(r'\d+', self.image_table_paths[ind])[0])
        table_index = self.number_to_index[number]

        return self.instance_norm(torch.from_numpy(self.feature_df.loc[self.all_IDs[ind]].values)), \
            torch.from_numpy(self.labels_df.loc[self.all_IDs[ind]].values), \
            self.numeric_input[table_index], \
            self.categorical_input[table_index], \
            batch['image']

    def __len__(self):
        return len(self.all_IDs)


if __name__ == '__main__':
    # data_set = UEAloader(
    #     root_path='../datasets/EEG/train/',
    #     table_path='../datasets/Table/train.xlsx',
    #     image_path='../datasets/Image/train/',
    #     flag='TRAIN',
    # )
    data_set = UEAloader(
        root_path='C:/Users/JustlfC/Downloads/MTSNet/datasets/EEG/train/',
        table_path='C:/Users/JustlfC/Downloads/MTSNet/datasets/Table/train.xlsx',
        image_path='C:/Users/JustlfC/Downloads/MTSNet/datasets/Image/train/',
        flag='TRAIN',
    )

    print(data_set.__len__())
    print(data_set[8])
