import pdb
import torch
import os
from torch import nn
import torch.nn.functional as F
from torchcrf import CRF
from .modeling_bert import BertModel
from .layers import CrossAttention
from transformers.modeling_outputs import TokenClassifierOutput
from torchvision.models import resnet50
from torch.autograd import Variable
from torch.distributions.kl import kl_divergence
from torch.distributions import Normal
from transformers import ViTFeatureExtractor, ViTModel
from .modeling_hzk2 import ContrastiveLossAdapter
from .img_extraction_module import Focus, Conv, BottleneckCSP, SPP, Concat
from .spatial_channel_attention import SpatialAttention, ChannelAttention

def kld_gauss(mean_1, logsigma_1, mean_2, logsigma_2):
    """Using std to compute KLD"""
    sigma_1 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_1, 0.4)))
    sigma_2 = torch.exp(0.1 + 0.9 * F.softplus(torch.clamp_max(logsigma_2, 0.4)))
    q_target = Normal(mean_1, sigma_1)
    q_context = Normal(mean_2, sigma_2)
    kl = kl_divergence(q_target, q_context).mean(dim=0).sum()
    return kl


def reparameters(mean, logstd, mode):
    sigma = torch.exp(0.5 * logstd)
    gaussian_noise = torch.randn(mean.shape).cuda(mean.device)
    # sampled_z = gaussian_noise * sigma + mean
    if mode == 'train':
        sampled_z = gaussian_noise * sigma + mean
    else:
        sampled_z = mean
    kdl_loss = -0.5 * torch.mean(1 + logstd - mean.pow(2) - logstd.exp())
    return sampled_z, kdl_loss

# class CSPResNetConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, channels):
#         super(CSPResNetConvBlock, self).__init__()
#         self.cbl = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels//2, kernel_size=3,stride=2, padding=1),
#             nn.BatchNorm2d(out_channels//2),
#             nn.LeakyReLU(inplace=True)
#         )
#         self.cbl1 = nn.Sequential(
#             nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels//2),
#             nn.LeakyReLU(inplace=True)
#         )
#         self.cbl2 = nn.Sequential(
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.LeakyReLU(inplace=True)
#         )
#
#         self.resnet_blocks = nn.Sequential(*[ResNetBlock(out_channels, out_channels) for _ in range(channels)])
#         self.bn = nn.BatchNorm2d(out_channels)
#         self.conv = nn.Conv2d(out_channels//2, out_channels//2, kernel_size=3, padding=1)
#         self.SiLU=nn.SiLU(inplace=True)
#     def forward(self, x):
#         x_prime = self.cbl(x)#
#         x1_prime = self.conv(x_prime)#
#
#         x1 = self.resnet_blocks(self.cbl1(x_prime))#
#         x2_prime = self.conv(x1)#3，64，224，224
#         x_prime = torch.cat((x1_prime, x2_prime), dim=1)  # 连接特征映射3
#         # 后续处理
#         x_prime = self.bn(x_prime)  # 批处理归一化
#         x_prime = self.SiLU(x_prime)  # SiLU 激活函数
#         y = self.cbl2(x_prime)  # CBL 层
#
#         return y
# class ResNetBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(ResNetBlock, self).__init__()
#         self.cbl1 = nn.Sequential(
#             nn.Conv2d(in_channels//2, out_channels//2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels //2),
#             nn.LeakyReLU(inplace=True)
#         )
#         self.cbl2 = nn.Sequential(
#             nn.Conv2d(in_channels//2, out_channels//2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(out_channels // 2),
#             nn.LeakyReLU(inplace=True)
#         )
#
#     def forward(self, x):
#         shortcut = x
#         x = self.cbl1(x)
#         x = self.cbl2(x)
#         x += shortcut
#         return x
# class TopDownBuildingBlock(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(TopDownBuildingBlock, self).__init__()
#         self.csp_resnet_conv_block = CSPResNetConvBlock(in_channels, out_channels,3)
#
#     def forward(self, coarse_representation, finer_representation):
#         # 上采样
#         upsampled_coarse_representation = nn.Upsample(scale_factor=2, mode='nearest')(coarse_representation)
#         # 横向连接
#         concatenated_representation = torch.cat((upsampled_coarse_representation, finer_representation), dim=1)
#         # 通过CSPResNet卷积块
#         output = self.csp_resnet_conv_block(concatenated_representation)
#         return output
#
# # 自上而下路径
# class TopDownPath(nn.Module):
#     def __init__(self):
#         super(TopDownPath, self).__init__()
#         # self.conv_block = CSPResNetConvBlock(3, 64,3)  # 假设输入图像为3通道
#         self.building_block1 = TopDownBuildingBlock(384, 512)
#         self.building_block2 = TopDownBuildingBlock(1024, 1024)
#
#     def forward(self, r1, r2, r3):
#         # g1 = self.conv_block(r1)
#         g2 = self.building_block1(r2, r1)
#         g3 = self.building_block2(r3, g2)
#         return r1, g2, g3
#
# # 自底向上路径
# class BottomUpPath(nn.Module):
#     def __init__(self):
#         super(BottomUpPath, self).__init__()
#         self.cbl = nn.Sequential(
#             nn.Conv2d(128, 512, kernel_size=1, stride=2),
#             nn.BatchNorm2d(512),
#             nn.LeakyReLU(inplace=True)
#         )
#         self.csp_resnet_block = CSPResNetConvBlock(1024, 1024,3)
#
#     def forward(self, g1, g2, g3):
#         # 下采样G1
#         downsampled_g1 = self.cbl(g1)
#         # 连接G1和G2
#         concatenated = torch.cat((downsampled_g1, g2), dim=1)
#         # 提取视觉信息
#         o = self.csp_resnet_block(concatenated)
#         # 积分得到F
#         f = torch.cat((o, g3), dim=1)
#         return f
#
# # 定义骨干网络
# class BackboneNetwork(nn.Module):
#     def __init__(self):
#         super(BackboneNetwork, self).__init__()
#         self.conv1 = CSPResNetConvBlock(3, 64,3)  # 假设输入图像为3通道
#         self.conv2 = CSPResNetConvBlock(64, 128,6)
#         self.conv3 = CSPResNetConvBlock(128, 256,9)
#         self.conv4 = CSPResNetConvBlock(256, 512,3)
#         # self.conv4 = CSPResNetConvBlock(1024, 1024)
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#         x2 = self.conv2(x1)
#         x3 = self.conv3(x2)
#         x4 = self.conv4(x3)
#         return x2, x3, x4
# class BidirectionalFeatureFusion(nn.Module):
#     def __init__(self):
#         super(BidirectionalFeatureFusion, self).__init__()
#         self.top_down_path = TopDownPath()
#         self.bottom_up_path = BottomUpPath()
#
#     def forward(self, r1, r2, r3):
#         # 自上而下路径
#         g1, g2, g3 = self.top_down_path(r1, r2, r3)
#
#         # 自下而上路径
#         f = self.bottom_up_path(g1, g2, g3)
#
#         return f
# class ImageModel(nn.Module):
#     def __init__(self):
#         super(ImageModel, self).__init__()
#         # Load a pretrained ResNet as the base model
#         # self.resnet = resnet50(pretrained=True)
#
#         # Replace the first layer with a CSPResNet block
#         # self.conv1 = CSPResNetBlock(3, 64)
#         self.conv1 = BackboneNetwork()
#         self.feature_fusion = BidirectionalFeatureFusion()
#         # self.csp_resnet_block = CSPResNetConvBlock(2048, 2048, 3)
#         # self.TextModulatedVisualAttention = TextModulatedVisualAttention()
#         self.bn1 = nn.BatchNorm2d(2048)
#         self.relu = nn.LeakyReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         # self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
#         # self.maxpool = nn.AdaptiveAvgPool2d((1, 1))
#     def forward(self, x, att_size=7):
#
#         r1, r2, r3 = self.conv1(x)
#         x = self.feature_fusion(r1, r2, r3)
#         # x = self.TextModulatedVisualAttention(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         # fc = x.mean(3).mean(2)
#         # att = F.adaptive_avg_pool2d(x,[att_size,att_size])
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#
#         # if not self.if_fine_tune:
#
#         x= Variable(x.data)
#         # fc = Variable(fc.data)
#         # att = Variable(att.data)
#
#         return x

class ImageModel(nn.Module):
    def __init__(self, hidden_size, negative_slope1=0.01, negative_slope2=0.01, k=10):
        super(ImageModel, self).__init__()
        self.hidden_size = hidden_size
        self.focus = Focus(c1=3, c2=64, k=3, negative_slope=negative_slope2)
        self.cbl_1 = Conv(c1=64, c2=128, k=3, s=2, negative_slope=negative_slope1)
        self.csp_1 = BottleneckCSP(c1=128, c2=128, n=3, negative_slope=negative_slope2)

        self.cbl_2 = Conv(c1=128, c2=256, k=3, s=2, negative_slope=negative_slope1)
        self.csp_2 = BottleneckCSP(c1=256, c2=256, n=6, negative_slope=negative_slope2)  # layer4

        self.cbl_3 = Conv(c1=256, c2=512, k=3, s=2, negative_slope=negative_slope1)
        self.csp_3 = BottleneckCSP(c1=512, c2=512, n=9, negative_slope=negative_slope2)  # layer6

        self.cbl_4 = Conv(c1=512, c2=1024, k=3, s=2, negative_slope=negative_slope1)
        self.csp_4 = BottleneckCSP(c1=1024, c2=1024, n=3, negative_slope=negative_slope2)
        self.ssp = SPP(c1=1024, c2=1024, k=[5, 9, 13], negative_slope=negative_slope2)

        # self.spatial_atten = SpatialAttention(in_planes=1024, text_dim=hidden_size, k=k)
        # self.channel_atten = ChannelAttention(in_planes=1024, text_dim=hidden_size, k=k)

        self.cbl_5 = Conv(c1=1024, c2=512, k=1, s=1, negative_slope=negative_slope1)
        self.upsample1 = nn.Upsample(size=None, scale_factor=2, mode='nearest')
        self.concat1 = Concat(1)
        self.csp_5 = BottleneckCSP(c1=512 + 512, c2=512, n=3, shortcut=False, negative_slope=negative_slope2)

        self.cbl_6 = Conv(c1=512, c2=256, k=1, s=1, negative_slope=negative_slope1)
        self.upsample2 = nn.Upsample(size=None, scale_factor=2, mode='nearest')
        self.concat2 = Concat(1)
        self.csp_6 = BottleneckCSP(c1=256 + 256, c2=256, n=3, shortcut=False, negative_slope=negative_slope2)

        self.cbl_7 = Conv(c1=256, c2=256, k=3, s=2, negative_slope=negative_slope1)
        self.concat3 = Concat(1)
        self.csp_7 = BottleneckCSP(c1=256 + 256, c2=512, n=3, shortcut=False, negative_slope=negative_slope2)

        self.cbl_8 = Conv(c1=512, c2=512, k=3, s=2, negative_slope=negative_slope1)
        self.concat4 = Concat(1)
        self.csp_8 = BottleneckCSP(c1=512 + 512, c2=1024, n=3, shortcut=False, negative_slope=negative_slope2)

        self.cbl_final = Conv(c1=1024, c2=self.hidden_size, k=1, s=1, negative_slope=negative_slope1)
        self.csp_final = BottleneckCSP(c1=self.hidden_size, c2=self.hidden_size, n=3, shortcut=True,
                                           negative_slope=negative_slope2)

        self.bn1 = nn.BatchNorm2d(1024)
        self.relu = nn.LeakyReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        layer4 = self.csp_2(self.cbl_2(self.csp_1(self.cbl_1(self.focus(x)))))
        layer6 = self.csp_3(self.cbl_3(layer4))
        layer9 = self.ssp(self.csp_4(self.cbl_4(layer6)))

        layer10 = self.cbl_5(layer9)  # G3
        layer13 = self.csp_5(self.concat1([self.upsample1(layer10), layer6]))

        layer14 = self.cbl_6(layer13)  # G2
        layer17 = self.csp_6(self.concat2([self.upsample2(layer14), layer4]))  # G1

        layer20 = self.csp_7(self.concat3([self.cbl_7(layer17), layer14]))

        layer23 = self.csp_8(self.concat4([self.cbl_8(layer20), layer10]))  # (32,1024,7,7)

            # layer23 = layer23.mul(self.channel_atten(layer23, text_feat))
            # layer23 = layer23.mul(self.spatial_atten(layer23, text_feat))
        x = self.bn1(layer23)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        final_layer = x.view(x.size(0), -1)
        # final_layer = Variable(final_layer.data)
        # final_layer = self.csp_final(self.cbl_final(layer23))
        return final_layer
# class ImageModel(nn.Module):
#     def __init__(self):
#         super(ImageModel, self).__init__()
#         self.resnet = resnet50(pretrained=True)
#         # self.if_fine_tune = if_fine_tune
#         # self.device = device
#
#     def forward(self, x, att_size=7):
#         x = self.resnet.conv1(x)
#         x = self.resnet.bn1(x)
#
#         x = self.resnet.relu(x)
#         x = self.resnet.maxpool(x)l1

#
#         x = self.resnet.layer1(x)
#         x = self.resnet.layer2(x)
#         x = self.resnet.layer3(x)
#         x = self.resnet.layer4(x)
#
#         fc = x.mean(3).mean(2)
#         att = F.adaptive_avg_pool2d(x,[att_size,att_size])
#
#         x = self.resnet.avgpool(x)
#         x = x.view(x.size(0), -1)
#
#         # if not self.if_fine_tune:
#
#         x= Variable(x.data)
#         fc = Variable(fc.data)
#         att = Variable(att.data)
#
#         return x, fc, att

class HVPNeTREModel(nn.Module):
    def __init__(self, num_labels, tokenizer, args):
        super(HVPNeTREModel, self).__init__()
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.args = args
        self.vis_encoding = ImageModel()
        self.hidden_size = args.hidden_size

        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(self.bert.config.hidden_size * 2, num_labels)
        self.head_start = tokenizer.convert_tokens_to_ids("<s>")
        self.tail_start = tokenizer.convert_tokens_to_ids("<o>")
        self.tokenizer = tokenizer
        self.linear = nn.Linear(2048, self.hidden_size)

        self.txt_encoding_mean = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                                dropout=args.dropout)
        self.txt_encoding_logstd = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                                  dropout=args.dropout)
        self.img_encoding_mean = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                                dropout=args.dropout)
        self.img_encoding_logstd = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                                  dropout=args.dropout)

        self.score_func = self.args.score_func
        if self.score_func == 'bilinear':
            self.discrimonator = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
        elif self.score_func == 'concat':
            self.discrimonator = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if args.fusion == 'cross':
            self.img2txt_cross = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                                dropout=args.dropout)
            self.txt2img_cross = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                                dropout=args.dropout)
        elif args.fusion == 'concat':
            self.cross_encoder = nn.Linear(self.hidden_size * 2, self.hidden_size)
        elif args.fusion == 'add':
            self.ln = nn.LayerNorm(self.hidden_size)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            labels=None,
            images=None,
            aux_imgs=None,
            mode='train'
    ):

        output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True)

        sequence_output, pooler_output = output.last_hidden_state, output.pooler_output
        batch_size, seq_len, hidden_size = sequence_output.shape

        all_images_ = torch.cat([images.unsqueeze(1), aux_imgs], dim=1)  # [batch, m+1, 3, 224, 224]
        all_images_rep_, _, att_all_images = self.vis_encoding(all_images_.reshape(-1, 3, 224, 224))
        all_images = all_images_rep_.reshape(-1, self.args.m + 1, 2048)  # [batch, m+1, 2048]
        all_images = self.linear(all_images)

        txt_mean = self.txt_encoding_mean(sequence_output, sequence_output, sequence_output,
                                          attention_mask.unsqueeze(1).unsqueeze(-1))
        txt_logstd = self.txt_encoding_logstd(sequence_output, sequence_output, sequence_output,
                                              attention_mask.unsqueeze(1).unsqueeze(-1))
        img_mean = self.img_encoding_mean(all_images, all_images, all_images, None)
        img_logstd = self.img_encoding_logstd(all_images, all_images, all_images, None)

        sample_z_txt, txt_kdl = reparameters(txt_mean, txt_logstd, mode)  # [batch, seq_len, dim]
        sample_z_img, img_kdl = reparameters(img_mean, img_logstd, mode)  # [batch, seq_len, dim]

        if self.args.reduction == 'mean':
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt.mean(dim=1), sample_z_img.mean(dim=1)
        elif self.args.reduction == 'sum':
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt.sum(dim=1), sample_z_img.sum(dim=1)
        else:
            sample_z_txt_cls, sample_z_img_cls = sample_z_txt[:, 0, :], sample_z_img[:, 0, :]
        if self.score_func == 'bilinear':
            pos_img_txt_score = torch.sigmoid(
                self.discrimonator(sample_z_txt_cls.unsqueeze(1), sample_z_img_cls.unsqueeze(1))).squeeze(1)
        elif self.score_func == 'concat':
            pos_img_txt_score = torch.sigmoid(
                self.discrimonator(torch.cat([sample_z_txt_cls, sample_z_img_cls], dim=-1)))
        pos_dis_loss = nn.functional.binary_cross_entropy(pos_img_txt_score, torch.ones(pos_img_txt_score.shape).to(
            pos_img_txt_score.device))

        neg_dis_loss = 0
        for s in range(1, self.args.neg_num + 1):
            neg_sample_z_img_cls = sample_z_img_cls.roll(shifts=s, dims=0)
            if self.score_func == 'bilinear':
                neg_img_txt_score = torch.sigmoid(
                    self.discrimonator(sample_z_txt_cls.unsqueeze(1), neg_sample_z_img_cls.unsqueeze(1))).squeeze(1)
            elif self.score_func == 'concat':
                neg_img_txt_score = torch.sigmoid(
                    self.discrimonator(torch.cat([sample_z_txt_cls, neg_sample_z_img_cls], dim=-1)))

            neg_dis_loss_ = nn.functional.binary_cross_entropy(neg_img_txt_score,
                                                               torch.zeros(neg_img_txt_score.shape).to(
                                                                   neg_img_txt_score.device))
            neg_dis_loss += neg_dis_loss_
        dis_loss = pos_dis_loss + neg_dis_loss

        out = self.img2txt_cross(sample_z_img, sample_z_txt, sample_z_txt, None)
        final_txt = self.txt2img_cross(sample_z_txt, out, out, attention_mask.unsqueeze(1).unsqueeze(-1))
        # pdb.set_trace()
        entity_hidden_state = torch.Tensor(batch_size, 2 * hidden_size)  # batch, 2*hidden
        for i in range(batch_size):
            head_idx = input_ids[i].eq(self.head_start).nonzero().item()
            tail_idx = input_ids[i].eq(self.tail_start).nonzero().item()
            head_hidden = final_txt[i, head_idx, :].squeeze()
            tail_hidden = final_txt[i, tail_idx, :].squeeze()
            entity_hidden_state[i] = torch.cat([head_hidden, tail_hidden], dim=-1)
        entity_hidden_state = entity_hidden_state.to(self.args.device)
        logits = self.classifier(entity_hidden_state)
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels.view(-1), reduction='sum')
            return loss, dis_loss, txt_kdl, img_kdl, logits
        return logits


class HVPNeTNERModel(nn.Module):
    def __init__(self, label_list, args):
        super(HVPNeTNERModel, self).__init__()
        self.args = args
        self.hidden_size = args.hidden_size
        self.bert = BertModel.from_pretrained(args.bert_name)
        self.bert_config = self.bert.config
        self.vis_encoding = ImageModel(self.bert.config.hidden_size,)
        self.num_labels = len(label_list)  # pad
        self.crf = CRF(self.num_labels, batch_first=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = nn.Dropout(args.dropout)
        self.drop = nn.Dropout(0.5)
        self.WC = nn.Linear(self.bert.config.hidden_size + self.hidden_size, self.hidden_size, bias=True)
        self.WI = nn.Linear(self.bert.config.hidden_size + self.hidden_size, self.hidden_size, bias=True)

        # 门控向量的激活函数
        self.act = nn.LeakyReLU(inplace=True)
        # self.act = nn.SiLU()
        self.temp = 0.5
        # 用于分离模态相关特征的权重矩阵
        self.WC_prime = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.WI_prime = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.linear = nn.Linear(1024, args.hidden_size, bias=True)

        self.txt_mean = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                       dropout=args.dropout)
        self.img_mean = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                       dropout=args.dropout)
        self.txt_logstd = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                         dropout=args.dropout)
        self.img_logstd = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                         dropout=args.dropout)
        self.contrastive_loss_adapter = ContrastiveLossAdapter()
        self.ScaledDotProductAttention = ScaledDotProductAttention(768)
        self.score_func = self.args.score_func
        if self.score_func == 'bilinear':
            self.discrimonator = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
        elif self.score_func == 'concat':
            self.discrimonator = nn.Linear(self.hidden_size * 2, self.hidden_size)

        if args.fusion == 'cross':
            self.img2txt_cross = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                                dropout=args.dropout)
            self.txt2img_cross = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                                dropout=args.dropout)
        elif args.fusion == 'concat':
            self.cross_encoder = nn.Linear(self.hidden_size * 2, self.hidden_size)
        elif args.fusion == 'add':
            self.ln = nn.LayerNorm(self.hidden_size)
        elif args.fusion == 'gate':
            self.img2txt_cross = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                                dropout=args.dropout)
            self.txt2img_cross = CrossAttention(heads=12, in_size=args.hidden_size, out_size=args.hidden_size,
                                                dropout=args.dropout)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, images=None, aux_imgs=None,
                images_dif=None, aux_images_dif=None, mode='train'):

        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(bert_output.last_hidden_state)  # bsz, len, hidden

        all_images_ = torch.cat([images.unsqueeze(1), aux_imgs], dim=1)  # [batch, m+1, 3, 224, 224]
        all_images_rep_ = self.vis_encoding(all_images_.reshape(-1, 3, 224, 224))
        all_images = all_images_rep_.reshape(-1, self.args.m + 1, 1024)  # [batch, m+1, 2048]
        all_images = self.linear(all_images)

        all_images_diff_ = torch.cat([images_dif.unsqueeze(1), aux_images_dif], dim=1)  # [batch, m+1, 3, 224, 224]
        all_images_diff_rep_ = self.vis_encoding(all_images_diff_.reshape(-1, 3, 224, 224))
        all_diff_images = all_images_diff_rep_.reshape(-1, 4, 1024)  # [batch, m+1, 2048]
        all_diff_images = self.linear(all_diff_images)

        txt_mean = self.txt_mean(sequence_output, sequence_output, sequence_output,
                                 attention_mask.unsqueeze(1).unsqueeze(-1))
        img_mean = self.img_mean(all_images, all_images, all_images, None)
        txt_logstd = self.txt_logstd(sequence_output, sequence_output, sequence_output,
                                     attention_mask.unsqueeze(1).unsqueeze(-1))
        img_logstd = self.img_logstd(all_images, all_images, all_images, None)

        img_diff_mean = self.img_mean(all_diff_images,  all_diff_images, all_diff_images, None)
        img__diff_logstd = self.img_logstd(all_diff_images,  all_diff_images, all_diff_images, None)

        sample_z_txt, txt_kdl = reparameters(txt_mean, txt_logstd, mode)  # [batch, seq_len, dim]
        sample_z_img, img_kdl = reparameters(img_mean, img_logstd, mode)  # [batch, seq_len, dim]

        sample_z_img_diff, img_kdl_diff = reparameters(img_diff_mean, img__diff_logstd, mode)  # [batch, seq_len, dim]

        # pooling操作
        if self.args.reduction == 'mean':
            sample_z_txt_cls, sample_z_img_cls, sample_z_img_diff_cls = sample_z_txt.mean(dim=1), sample_z_img.mean(
                dim=1), sample_z_img_diff.mean(dim=1)

        elif self.args.reduction == 'sum':
            sample_z_txt_cls, sample_z_img_cls, sample_z_img_diff_cls = sample_z_txt.sum(dim=1), sample_z_img.sum(
                dim=1), sample_z_img_diff.sum(dim=1)
        else:
            sample_z_txt_cls, sample_z_img_cls, sample_z_img_diff_cls = sample_z_txt[:, 0, :], sample_z_img[:, 0,:], sample_z_img_diff[:, 0, :]


        if self.score_func == 'bilinear':
            pos_img_txt_score = torch.sigmoid(
                self.discrimonator(sample_z_txt_cls.unsqueeze(1), sample_z_img_cls.unsqueeze(1))).squeeze(1)
            pos_diff_img_txt_score = torch.sigmoid(
                self.discrimonator(sample_z_txt_cls.unsqueeze(1), sample_z_img_diff_cls.unsqueeze(1))).squeeze(1)
        elif self.score_func == 'concat':
            pos_img_txt_score = torch.sigmoid(
                self.discrimonator(torch.cat([sample_z_txt_cls, sample_z_img_cls], dim=-1)))
            pos_diff_img_txt_score = torch.sigmoid(
                self.discrimonator(torch.cat([sample_z_txt_cls, sample_z_img_diff_cls], dim=-1)))

        pos_dis_loss = nn.functional.binary_cross_entropy(pos_img_txt_score, torch.ones(pos_img_txt_score.shape).to(
            pos_img_txt_score.device))
        pos_dis_diff_loss = nn.functional.binary_cross_entropy(pos_diff_img_txt_score,
                                                               torch.ones(pos_diff_img_txt_score.shape).to(
                                                                   pos_diff_img_txt_score.device))
        # D判别器D(z^T,Z^V-) neg_img_txt_score
        neg_dis_loss = 0
        neg_dis_diff_loss = 0
        for s in range(1, self.args.neg_num + 1):
            neg_sample_z_img_cls = sample_z_img_cls.roll(shifts=s, dims=0)
            neg_sample_z_img_diff__cls = sample_z_img_diff_cls.roll(shifts=s, dims=0)
            if self.score_func == 'bilinear':
                neg_img_txt_score = torch.sigmoid(
                    self.discrimonator(sample_z_txt_cls.unsqueeze(1), neg_sample_z_img_cls.unsqueeze(1))).squeeze(1)
                neg_diff_img_txt_score = torch.sigmoid(
                    self.discrimonator(sample_z_txt_cls.unsqueeze(1), neg_sample_z_img_diff__cls.unsqueeze(1))).squeeze(
                    1)
            elif self.score_func == 'concat':
                neg_img_txt_score = torch.sigmoid(
                    self.discrimonator(torch.cat([sample_z_txt_cls, neg_sample_z_img_cls], dim=-1)))
                neg_diff_img_txt_score = torch.sigmoid(
                    self.discrimonator(torch.cat([sample_z_txt_cls, neg_sample_z_img_diff__cls], dim=-1)))
            # neg_img_txt_score = torch.sigmoid(self.discrimonator(sample_z_txt_cls.unsqueeze(1), neg_sample_z_img_cls.unsqueeze(1))).squeeze(1)
            neg_dis_loss_ = nn.functional.binary_cross_entropy(neg_img_txt_score,
                                                               torch.zeros(neg_img_txt_score.shape).to(
                                                                   neg_img_txt_score.device))
            neg_dis_loss += neg_dis_loss_
            # neg_dis_diff_loss_ = nn.functional.binary_cross_entropy(neg_diff_img_txt_score,
            #                                                         torch.zeros(neg_diff_img_txt_score.shape).to(
            #                                                             neg_diff_img_txt_score.device))
            # neg_dis_diff_loss += neg_dis_diff_loss_
        dis_loss = pos_dis_loss + neg_dis_loss  # 判别器loss
        dis_diff_loss = pos_dis_diff_loss + neg_dis_diff_loss  # 判别器loss
        # dis_loss_ = dis_diff_loss + dis_loss
        # cl_loss = cl_loss1 + cl_loss
        kdl = txt_kdl + img_kdl + img_kdl_diff
        # kdl = txt_kdl + img_kdl
        final_txt = None  # 添加默认值
        z_img_cls=None
        z_img_diff_cls =None
        # Modality Fusion
        if self.args.fusion == 'cross':

            last_hidden_state_t, _ = self.ScaledDotProductAttention(sample_z_txt, sample_z_img,
                                                                    sample_z_img)

            final_txt = last_hidden_state_t
        if self.args.fusion == 'gate':

            final_txt1,  attn_weights1 = self.ScaledDotProductAttention(sample_z_txt, sample_z_img, sample_z_img)
            final_txt2,  attn_weights2 = self.ScaledDotProductAttention(sample_z_txt, sample_z_img_diff, sample_z_img_diff)

            gC = self.act(self.WC(torch.cat([final_txt1, sample_z_txt], dim=-1)))
            gI = self.act(self.WI(torch.cat([final_txt2, sample_z_txt], dim=-1)))
            H_C = gC * self.WC_prime(final_txt1)
            H_I = gI * self.WI_prime(final_txt2)


            H = H_C + H_I
            norm_ST = torch.norm(sample_z_txt,p=2, dim=(-2, -1), keepdim=True)
            norm_H = torch.norm(H, p=2,dim=(-2, -1), keepdim=True)
            beta_tensor = torch.tensor(self.args.beta, device=sample_z_txt.device, dtype=sample_z_txt.dtype)
            scaling_factor = torch.min((norm_ST / norm_H) * beta_tensor,
                                       torch.tensor(1.0, device=sample_z_txt.device, dtype=sample_z_txt.dtype))
            final = sample_z_txt + scaling_factor * H
            final_txt = self.drop(final)

        emissions = self.fc(final_txt)  # bsz, len, labels
        logits = self.crf.decode(emissions, attention_mask.byte())
        loss = None
        if labels is not None:
            loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte(), reduction='mean')
            # other_loss = dis_loss_ + txt_kdl + img_kdl + img_kdl_diff
        
        return TokenClassifierOutput(loss=loss, logits=logits), dis_loss, kdl


class ScaledDotProductAttention(nn.Module):
    def __init__(self, embed_dim, dropout=0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        self.scale = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float32))
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 1)
        self.fc3 = nn.Linear(61, 80)
        self.tanh = nn.Tanh()

    def forward(self, q, k, v):
        # 求g
        g_fc1 = self.fc1(v)
        g_tanh = self.tanh(g_fc1)
        g_fc2 = self.fc2(g_tanh)
        g = g_fc2.squeeze()
        # g2 = self.fc3(g1)
        # qk
        scores = torch.matmul(q, k.transpose(-1, -2))
        # 求s
        # 对注意力分数进行缩放处理
        scores_scaled = scores / torch.sqrt(torch.tensor(q.shape[1]).float())
        # 对注意力分数进行softmax归一化
        s = F.softmax(scores_scaled, dim=1)#to
        # weights
        # torch.Size([80, 61]) torch.Size([61]) torch.Size([80, 61]) torch.Size([61, 1])
        # print(torch.mean(s, dim=1, keepdim=True).size())
        # print(scores_scaled.size(), g.size(), s.size(), g_fc2.size())
        c=torch.mean(s, dim=1, keepdim=True)
        g = g.unsqueeze(-2)
        attn_scores = torch.add(torch.mul(s, (1 - g)), torch.mul(torch.mean(s, dim=1, keepdim=True), g)) / self.scale
        attn_weights = self.softmax(attn_scores)
        attn_weights = self.dropout(attn_weights)
        # output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output + q
        # attn_output = self.dropout(attn_output)
        return attn_output, attn_weights