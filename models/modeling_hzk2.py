import numpy as np
import random

import torch
import torch.nn as nn
from torch.autograd import Function
from transformers import BertModel, BertConfig

from .hzk_utils import to_gpu
from .hzk_utils import ReverseLayerF

# clip
from .CLIP import *

from typing import Any, Optional, Tuple

# loss
from .hzk_utils import to_gpu, time_desc_decorator, DiffLoss, MSE, \
    SIMSE, CMD,SIM


class HZK2(nn.Module):

    def __init__(self, config):
        super(HZK2, self).__init__()
        # config,size
        self.config = config
        # 0 self.text_size = config.embedding_size
        self.text_size = 768
        # 0 self.visual_size = config.visual_size
        self.visual_size = 768

        # size
        self.input_sizes = input_sizes = [self.text_size, self.visual_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size)]
        # self.output_size = output_size = config.num_classes
        self.output_size = output_size = 23
        # dropout
        # self.dropout_rate = dropout_rate = config.dropout
        self.dropout_rate = dropout_rate = 0.2

        # activation
        # self.activation = self.config.activation()
        self.activation = nn.Sigmoid()
        self.tanh = nn.Tanh()

        # #! bert-t
        # bertconfig = BertConfig.from_pretrained(self.config.bert_name, output_hidden_states=True)
        # self.text_config=bertconfig
        # self.text_embeddings = BertEmbeddings(bertconfig)
        # self.bertmodel = BertModel.from_pretrained(self.config.bert_name, config=bertconfig)
        # #! IFA vision model
        # self.vision_config = vision_config
        # self.device = vision_config.device
        # self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        # self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        # self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)

        # project_t
        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v',
                                  nn.Linear(in_features=hidden_sizes[1], out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))
        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())

        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())
        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())
        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1',
                                          nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2',
                                          nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1',
                                         nn.Linear(in_features=config.hidden_size, out_features=4))
        # fusion
        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size * 4,
                                                           out_features=self.config.hidden_size * 2))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=self.config.hidden_size * 2, out_features=output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))

        # transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        #     loss
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()
        self.loss_cmd2 = SIM()

        self.contrastive_loss_adapter=ContrastiveLossAdapter()

    def shared_private(self, utterance_t, utterance_v):
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)

    def shared_private2(self, utterance_t, utterance_v):
        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)

        return torch.add(self.utt_private_t, self.utt_shared_t),torch.add(self.utt_private_v, self.utt_shared_v)
    
    def reconstruct(self, ):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)

    def forward(self, vision_embedding_output, text_embedding_output, lengths=None,
                inputs=None, labels=None,
                return_dict=None, ):

        utterance_text, utterance_video = text_embedding_output, vision_embedding_output
        # 返回
        T_recon, V_recon=self.shared_private2(utterance_text, utterance_video)

        contrastive_loss=self.contrastive_loss_adapter(self.utt_shared_t, self.utt_shared_v)

        # 压缩
        self.utt_shared_t = torch.mean(self.utt_shared_t, dim=1, keepdim=False)
        self.utt_shared_v = torch.mean(self.utt_shared_v, dim=1, keepdim=False)
        self.utt_private_t= torch.mean(self.utt_private_t, dim=1, keepdim=False)
        self.utt_private_v = torch.mean(self.utt_private_v, dim=1, keepdim=False)
        self.utt_t_orig = torch.mean(self.utt_t_orig, dim=1, keepdim=False)
        self.utt_v_orig = torch.mean(self.utt_v_orig, dim=1, keepdim=False)


        # if not self.config.use_cmd_sim:
        #     # discriminator
        #     reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
        #     reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)

        #     self.domain_label_t = self.discriminator(reversed_shared_code_t)
        #     self.domain_label_v = self.discriminator(reversed_shared_code_v)

        # else:
        #     self.domain_label_t = None
        #     self.domain_label_v = None
        #     self.domain_label_a = None

        # self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        # self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)

        # self.shared_or_private_s = self.sp_discriminator(
        #     (self.utt_shared_t + self.utt_shared_v) / 2.0)

        # For reconstruction
        self.reconstruct()

        # torch.Size([32, 80, 768]) torch.Size([32, 1, 768]) torch.Size([32, 768])
        # print(T_recon.size(), self.utt_shared_t.unsqueeze(1).size(), self.utt_private_t.size())
        # print(V_recon.size(), self.utt_shared_v.unsqueeze(1).size(), self.utt_private_v.size())
        T_recon, V_recon = T_recon+self.utt_shared_t.unsqueeze(1)+self.utt_private_t.unsqueeze(1), V_recon+self.utt_shared_v.unsqueeze(1)+self.utt_private_v.unsqueeze(1)

        return self.config.cmd_weight * contrastive_loss, self.config.diff_weight * self.get_diff_loss() ,self.config.recon_weight*self.get_recon_loss(),T_recon,V_recon

        # return o

    def get_diff_loss(self):

        # Between private and shared
        loss = self.loss_diff(self.utt_private_t, self.utt_shared_t)
        loss += self.loss_diff(self.utt_private_v, self.utt_shared_v)

        # Across privates

        loss += self.loss_diff(self.utt_private_t, self.utt_private_v)

        return loss

    def get_recon_loss(self, ):

        loss = self.loss_recon(self.utt_t_recon, self.utt_t_orig)
        loss += self.loss_recon(self.utt_v_recon, self.utt_v_orig)

        loss = loss / 2.0
        return loss

    def get_cmd_loss(self, ):

        # losses between shared states
        loss = self.loss_cmd(self.utt_shared_t, self.utt_shared_v, 5)

        return loss

    def get_cmd_loss2(self, ):

        # losses between shared states
        loss = self.loss_cmd2(self.utt_shared_t, self.utt_shared_v, 5)

        return loss

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)

    def get_input_embeddings(self):
        return self.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.text_embeddings.word_embeddings = value

    def _get_resized_embeddings(
            self, old_embeddings: nn.Embedding, new_num_tokens: Optional[int] = None
    ) -> nn.Embedding:
        """
        Build a resized Embedding Module from a provided token Embedding Module. Increasing the size will add newly
        initialized vectors at the end. Reducing the size will remove vectors from the end

        Args:
            old_embeddings (:obj:`torch.nn.Embedding`):
                Old embeddings to be resized.
            new_num_tokens (:obj:`int`, `optional`):
                New number of tokens in the embedding matrix.

                Increasing the size will add newly initialized vectors at the end. Reducing the size will remove
                vectors from the end. If not provided or :obj:`None`, just returns a pointer to the input tokens
                :obj:`torch.nn.Embedding`` module of the model without doing anything.

        Return:
            :obj:`torch.nn.Embedding`: Pointer to the resized Embedding Module or the old Embedding Module if
            :obj:`new_num_tokens` is :obj:`None`
        """
        if new_num_tokens is None:
            return old_embeddings
        else:
            old_num_tokens, old_embedding_dim = old_embeddings.weight.size()

        if old_num_tokens == new_num_tokens:
            return old_embeddings

        if not isinstance(old_embeddings, nn.Embedding):
            raise TypeError(
                f"Old embeddings are of type {type(old_embeddings)}, which is not an instance of {nn.Embedding}."
                f"You should either use a different resize function or make sure that `old_embeddings` are an instance of {nn.Embedding}."
            )

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim).to(
            self.device, dtype=old_embeddings.weight.dtype
        )

        # initialize all new embeddings (in particular added tokens)
        self._init_text_weights(new_embeddings)

        # Copy token embeddings from the previous weights

        # numbers of tokens to copy
        n = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:n, :] = old_embeddings.weight.data[:n, :]

        return new_embeddings

class ContrastiveLossAdapter1(nn.Module):
    def __init__(self,temperature=0.179):
        super().__init__()
        self.temperature=temperature
        self.fc1 = nn.Linear(4,1)
        self.fc2 = nn.Linear(4,1)
        self.gelu= nn.GELU()

    def forward(self,inputT,inputV):
        batch_size=inputV.shape[0]

        # print(inputT.size(), inputV.size())
        emb_V=self.gelu(self.fc1(inputV.transpose(1,2)).squeeze(2))
        emb_T=self.gelu(self.fc2(inputT.transpose(1,2)).squeeze(2))
        negatives_mask = torch.eye(batch_size, batch_size, dtype=bool).to("cuda")

        z_V = F.normalize(emb_V, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_T = F.normalize(emb_T, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_V, z_T], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_VT = torch.diag(similarity_matrix, batch_size)  # bs
        arf=F.softmax(sim_VT,dim=0)
        simij = similarity_matrix[0:batch_size,batch_size:]
        positive_mask=(negatives_mask==False)
        nominator = torch.exp(sim_VT / self.temperature)  # bs

        # print(sim_VT, negatives_mask, nominator)
        denominator = (1-arf)*negatives_mask*torch.exp(simij/ self.temperature)+positive_mask*torch.exp(similarity_matrix[0:batch_size,batch_size:]/ self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # bs
        loss = torch.sum(loss_partial) / (batch_size)
        return loss

class ContrastiveLossAdapter(nn.Module):
    def __init__(self,temperature=0.25):
        super().__init__()
        self.temperature=temperature
        self.fc1 = nn.Linear(4,1)
        self.fc2 = nn.Linear(128,1)
        self.gelu= nn.GELU()

    def forward(self,inputT,inputV):
        batch_size=inputV.shape[0]

        # print(inputT.size(), inputV.size())
        emb_V=self.gelu(self.fc1(inputV.transpose(1,2)).squeeze(2))
        emb_T=self.gelu(self.fc2(inputT.transpose(1,2)).squeeze(2))
        negatives_mask = torch.eye(batch_size, batch_size, dtype=bool).to("cuda")

        z_V = F.normalize(emb_V, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_T = F.normalize(emb_T, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_V, z_T], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0),
                                                dim=2)  # simi_mat: (2*bs, 2*bs)

        sim_VT = torch.diag(similarity_matrix, batch_size)  # bs
        arf=F.softmax(sim_VT,dim=0)
        simij = similarity_matrix[0:batch_size,batch_size:]
        positive_mask=(negatives_mask==False)
        nominator = torch.exp(sim_VT / self.temperature)  # bs

        # print(sim_VT, negatives_mask, nominator)
        denominator = (1-arf)*negatives_mask*torch.exp(simij/ self.temperature)+positive_mask*torch.exp(similarity_matrix[0:batch_size,batch_size:]/ self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # bs
        loss = torch.sum(loss_partial) / (batch_size)
        return loss

    def _init_text_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.text_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

        self.aux_position_embedding = nn.Embedding(48, self.embed_dim)
        self.register_buffer("aux_position_ids", torch.arange(48).expand((1, -1)))

        self.rcnn_position_embedding = nn.Embedding(12, self.embed_dim)
        self.register_buffer("rcnn_position_ids", torch.arange(12).expand((1, -1)))


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(
            self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length: seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



