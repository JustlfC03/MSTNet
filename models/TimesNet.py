import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1
from tableformer import ft_transformer


# Residual dense block
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate, norm='None'):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate, norm=norm))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv3d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)

        out = self.conv_1x1(out)

        out = out + x  # Residual
        return out


# Dense Block
class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, norm='None'):
        super(make_dense, self).__init__()
        self.conv = nn.Conv3d(nChannels, growthRate, kernel_size=3, padding=1, bias=False)
        self.norm = norm
        self.bn = nn.BatchNorm3d(growthRate)

    def forward(self, x):
        out = self.conv(x)
        if self.norm == 'BN':
            out = self.bn(out)
        out = F.relu(out)

        out = torch.cat((x, out), 1)
        return out


# Image Encoder module
class ImageEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ImageEncoder, self).__init__()
        self.conv_in_c1 = nn.Conv3d(input_dim, hidden_dim, kernel_size=3, padding=1, bias=True)
        self.bn_in_c1 = nn.BatchNorm3d(hidden_dim)
        self.RDB1_c1 = RDB(hidden_dim, nDenselayer=4, growthRate=32, norm='BN')
        self.RDB2_c1 = RDB(hidden_dim, nDenselayer=4, growthRate=32, norm='BN')
        self.RDB3_c1 = RDB(hidden_dim, nDenselayer=4, growthRate=32, norm='BN')

    def forward(self, image_input):
        x = F.relu(self.bn_in_c1(self.conv_in_c1(image_input)))
        x = self.RDB1_c1(x)
        x = F.avg_pool3d(x, 2)
        x = self.RDB2_c1(x)
        x = F.avg_pool3d(x, 2)
        x = self.RDB3_c1(x)
        x = F.avg_pool3d(x, 2)
        return x


#  Clinical Encoder module
# class ClinicalEncoder(nn.Module):
#     def __init__(self, numeric_input_dim, categorical_input_dim, hidden_dim):
#         super(ClinicalEncoder, self).__init__()
#         # Define 1D convolutional layers for clinical encoding
#         # hidden_dim = hidden_dim // 2
#         hidden_dim = hidden_dim // 4
#         self.categorical_conv = nn.Conv1d(categorical_input_dim, hidden_dim, kernel_size=1, stride=1)
#         self.categorical_bn = nn.BatchNorm1d(hidden_dim)
#         self.numerical_conv = nn.Conv1d(numeric_input_dim, hidden_dim, kernel_size=1, stride=1)
#         self.numerical_bn = nn.BatchNorm1d(hidden_dim)
#         self.activation = nn.SiLU()
#
#     def forward(self, numeric_input, cateforical_input, image_features):
#         # Apply convolutional layers with SiLU activation
#         numeric_embed = self.numerical_conv(numeric_input)
#         categorical_embed = self.categorical_conv(cateforical_input)
#
#         categorical_output = self.activation(self.categorical_bn(categorical_embed))
#         numerical_output = self.activation(self.numerical_bn(numeric_embed))
#         # print(categorical_output.shape)
#         # print(numerical_output.shape)
#
#         # get image input shape
#         _, _, D, H, W = image_features.shape
#
#         # reshape here
#         categorical_output = categorical_output[..., None, None].repeat(1, 1, D, H, W)
#         numerical_output = numerical_output[..., None, None].repeat(1, 1, D, H, W)
#         # print(categorical_output.shape)
#         # print(numerical_output.shape)
#
#         return categorical_output, numerical_output


# Visual-Text Self-Attention module


class SpatialTemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SpatialTemporalAttention, self).__init__()
        #  self-attention mechanism
        self.q_conv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        self.k_conv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        self.v_conv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        self.attention_conv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        self.output_conv = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1, stride=1)
        self.activation = nn.ReLU()

    def forward(self, eeg_input, image_input):
        # Apply convolutional operations to obtain query, key, and value vectors
        q = self.q_conv(eeg_input)
        k = self.k_conv(image_input)
        v = self.v_conv(image_input)
        # Compute attention scores and apply softmax
        attn_scores = F.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1)
        # Weighted sum of values based on attention scores
        weighted_values = torch.matmul(attn_scores, v)
        # Apply attention and output convolutions
        attention_output = self.activation(self.attention_conv(weighted_values))
        output = self.activation(self.output_conv(attention_output))
        return output


#  Multimodal Aggregator module
class MultimodalAggregator(nn.Module):
    def __init__(self, hidden_dim):
        super(MultimodalAggregator, self).__init__()
        # aggregation layer
        self.fc = nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=1)
        # Residual Dense Connection
        self.RDB_comb = RDB(hidden_dim * 2, nDenselayer=4, growthRate=64, norm='BN')
        self.conv1_comb = nn.Conv3d(hidden_dim * 2, hidden_dim, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_comb = nn.BatchNorm3d(hidden_dim)
        self.activation = nn.ReLU()
        self.conv2_comb = nn.Conv3d(hidden_dim, 16, kernel_size=3, padding=1, bias=True)

    def forward(self, f_vom, f_hat_vom):
        # linear layer with ReLU activation
        f_hat_vom = F.relu(self.fc(f_hat_vom))
        comb_RDB = self.RDB_comb(f_hat_vom + f_vom)
        new_agg = self.activation(self.bn1_comb(self.conv1_comb(comb_RDB)))
        final_agg = self.conv2_comb(new_agg)
        # print(final_agg.shape)
        return final_agg
        # return f_hat_vom + f_vom


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(configs.d_model, configs.d_ff,
                               num_kernels=configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(configs.d_ff, configs.d_model,
                               num_kernels=configs.num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class EEGFeatureExtractor(nn.Module):
    def __init__(self):
        super(EEGFeatureExtractor, self).__init__()

        # 1D 卷积层与池化层
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2,
                               padding=1)  # 输出形状: (3, 64, 23040)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 输出形状: (3, 64, 11520)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2,
                               padding=1)  # 输出形状: (3, 128, 5760)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 输出形状: (3, 128, 2880)

        self.conv3 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=2,
                               padding=1)  # 输出形状: (3, 64, 1440)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 输出形状: (3, 64, 720)

        self.conv4 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1)  # 输出形状: (3, 32, 360)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)  # 输出形状: (3, 32, 180)

        # 使用线性层将特征扩展到512
        self.fc = nn.Linear(180 * 32, 512 * 32)  # 输入大小为 32 * 180，输出大小为 32 * 512

    def forward(self, x):
        x = self.conv1(x)  # (3, 64, 23040)
        x = self.pool1(x)  # (3, 64, 11520)

        x = self.conv2(x)  # (3, 128, 5760)
        x = self.pool2(x)  # (3, 128, 2880)

        x = self.conv3(x)  # (3, 64, 1440)
        x = self.pool3(x)  # (3, 64, 720)

        x = self.conv4(x)  # (3, 32, 360)
        x = self.pool4(x)  # (3, 32, 180)

        # 展平并通过全连接层
        x = x.view(x.size(0), -1)  # 展开为 (3, 32*180)
        x = self.fc(x)  # 输出形状: (3, 32*512)

        return x.view(-1, 32, 512)  # 重塑为 (3, 32, 512)


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([TimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)

            # self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)
            # Full-Connected layers
            self.projection1 = nn.Linear(16 * 32 * 4 * 4, 1024, bias=True)
            self.projection2 = nn.Linear(1024, 128, bias=True)
            self.projection3 = nn.Linear(128, 16, bias=True)
            self.projection4 = nn.Linear(32, configs.num_class, bias=True) # img+eeg

        # image encoder and clinical encoder
        self.image_encoder = ImageEncoder(input_dim=configs.image_input_dim[0], hidden_dim=configs.hidden_dim)

        self.clinical_encoder = ft_transformer.FTTransformer(
            categories=configs.num_cat,  # tuple containing the number of unique values within each category
            num_continuous=configs.num_cont,  # number of continuous values
            dim=32,  # dimension, paper set at 32
            dim_out=16,  # binary prediction, but could be anything
            depth=6,  # depth, paper recommended 6
            heads=8,  # heads, paper recommends 8
            attn_dropout=0.1,  # post-attention dropout
            ff_dropout=0.1  # feed forward dropout
        )
        # 修改：
        # self.clinical_encoder = ClinicalEncoder(numeric_input_dim=configs.numeric_input_dim,
        #                                         categorical_input_dim=configs.categorical_input_dim,
        #                                         hidden_dim=configs.hidden_dim)

        self.spatial_temporal_attention = SpatialTemporalAttention(hidden_dim=configs.hidden_dim)
        self.aggregator = MultimodalAggregator(hidden_dim=configs.hidden_dim)

        self.eeg_model = EEGFeatureExtractor()

    def classification(self, x_enc, numeric_input, categorical_input, image_input, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)

        # zero-out padding embeddings
        # print("output: ", output.shape)
        # print("x_mark_enc: ", x_mark_enc.shape)
        output = output * x_mark_enc.unsqueeze(-1)  # (3, 4, 32)
        # print("output * x_mark_enc.unsqueeze(-1): ", output.shape)

        # (batch_size, seq_length * d_model)
        # output = output.reshape(output.shape[0], -1)
        # output = self.projection(output)  # (batch_size, num_classes)

        # image encoder
        image_features = self.image_encoder(image_input)  # (3, 256, 32, 4, 4)

        _, output_shape_seq_length, output_shape_d_model = output.shape  # (3, 46080, 32)
        _, C, D, H, W = image_features.shape
        new_output = output.transpose(2, 1)  # (3, 32, 46080)
        new_output = self.eeg_model(new_output)  # (3, 32, 512)
        new_output = new_output.view(-1, 32, 32, 4, 4)

        new_output = new_output.repeat(1, C // output_shape_d_model, 1, 1, 1)  # (3, 256, 32, 4, 4)
        # output = output[..., None, None].repeat(1, (C // 2) // output_shape_seq_length, D // output_shape_d_model, H, W)

        f_vom = torch.cat((new_output,
                           image_features),
                          dim=1)  # (3, 512, 32, 4, 4)

        # Apply visual-text attention mechanism(img+eeg)
        f_hat_vom = self.spatial_temporal_attention(new_output, image_features)  # (3, 256, 32, 4, 4)

        # Aggregate features from different modalities(img+eeg)
        aggregated_features = self.aggregator(f_vom, f_hat_vom)  # (3, 16, 32, 4, 4)

        img_eeg_features = aggregated_features.reshape(aggregated_features.shape[0], -1)
        # print(img_eeg_features.shape)

        # clinical encoder
        table_feature = self.clinical_encoder(categorical_input, numeric_input)
        # 修改：
        # categorical_output, numerical_output = self.clinical_encoder(numeric_input, categorical_input, image_features)

        img_eeg_features = self.projection1(img_eeg_features)  # (16 * 32 * 4 * 4)
        img_eeg_features = self.projection2(img_eeg_features)  # (1024, 128)
        img_eeg_features = self.projection3(img_eeg_features)  # (128, 16)

        # print(img_eeg_features.shape)
        # print(type(img_eeg_features))
        # print(img_eeg_features)
        stand_img_eeg_features = torch.tensor(img_eeg_features.data.cpu().numpy(), device=img_eeg_features.device)
        # print(table_feature.shape)
        # print(type(table_feature))
        # print(table_feature)
        final_features = torch.cat((stand_img_eeg_features, table_feature), dim=1)  # 在列方向拼接
        # print(final_features.shape)
        final_features = self.projection4(final_features)  # (batch_size, num_classes)

        return final_features

    def forward(self, x_enc, numeric_input, categorical_input, image_input, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, numeric_input, categorical_input, image_input, x_mark_enc)
            return dec_out  # [B, N]
        return None
