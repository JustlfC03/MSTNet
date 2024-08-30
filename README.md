# MSTNet

Overall structure of our proposed MSTNet. The MSTNet model primarily comprises three parts: Tabular Feature Encoder, Temporal Feature Encoder and Cross-modal Aggregation Encoder.
![image](img/model.png)

(a) The Feature Tokenizer converts numerical categorical features into embedding vectors. (b) The TimesBlock module performs a 2D transformation on the multi-periodic features of the time-series data.
![Comparison](img/module.png)

## 1. Dataset

- Place the dataset in the main folder with the following folder structure for the dataset：
`````
datasets/
├── Table/
│   ├── train.xlsx
│   └── test.xlsx
├── Image/
│   ├── train/
│   └── test/
└── EEG/
    ├── train/
    │   ├── MCI/
    │   ├── HC/
    │   └── AD/
    └── test/
        ├── MCI/
        ├── HC/
        └── AD/
`````
- The file names of the EEG and MRI of the same patient need to be consistent, while the Scale/Table information of all patients is saved in a `.xslx` file, and the first column `number` in the `.xslx` file also needs to be consistent with the corresponding EEG/MRI file name.

## 2. Environment

- Please prepare an environment with `python=3.8`, and then use the command `pip install -r requirements.txt` for the dependencies.

## 3. Train/Test

- Run run.py to Train or test (Put the MRI dataset in nii.gz format into datasets/Image and the EEG dataset in .csv format into datasets/EEG)
- The batch size we used is 20. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

## 4. ADMC dataset

- Our experiments were conducted on our private ADMC dataset, which comprises EEG, MRI, and Table data from 100 subjects (mean age: 72.4 years; age range: 56-93 years; 56 females; 22 married). For each patient, the dataset includes a continuous 180-second artifact-free EEG segment recorded at a sampling frequency of 256 Hz, and 72 MRI slices with dimensions of 256×256 pixels. Additionally, the Table data includes MMSE and MoCA scores, with detailed MMSE results providing both a total score and individual item scores. The dataset is divided into 80 samples for training and 20 samples for evaluation.

## 5. Citation

```
@misc{chen2024robustearlydetectionalzheimers,
      title={Toward Robust Early Detection of Alzheimer's Disease via an Integrated Multimodal Learning Approach}, 
      author={Yifei Chen and Shenghao Zhu and Zhaojie Fang and Chang Liu and Binfeng Zou and Yuhe Wang and Shuo Chang and Fan Jia and Feiwei Qin and Jin Fan and Yong Peng and Changmiao Wang},
      year={2024},
      eprint={2408.16343},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.16343}, 
}
```
