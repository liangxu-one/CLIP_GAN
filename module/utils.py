import torch, copy, math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from typing import Optional, Any, Union, Callable
from torch import Tensor

def _sample(model, img, caption_index, config):
    isFin = torch.ones_like(caption_index).reshape(-1)
    isUnFin = torch.zeros_like(caption_index).reshape(-1)
    eos_index = torch.tensor([config.eos_token_id]).to(img.device)

    sum_log = torch.zeros(img.size(0)).to(img.device)
    # 计算一次图片编码, 加快解码速度
    img_embed = model(img)
    for i in range(config.generation_length):

        # 若某个句子已经到达结束符, 将其状态设置为已完成
        last_token = caption_index[:, -1]
        flag = torch.where(last_token == eos_index, isFin, isUnFin)

        if torch.sum(flag) == torch.sum(isFin):
            break

        caption_mask = config.generator_fun(caption_index.size(1)).to(img.device).unsqueeze(0).repeat(caption_index.size(0), 1, 1)
        pred = model(img, caption_index, caption_mask, img_embed = img_embed)
        next = pred[:, -1, :]

        # 蒙特卡洛采样
        score = next.softmax(dim = -1)
        sample_index = torch.multinomial(score, 1)
        # 取出采样概率
        logits = next.log_softmax(dim = -1)
        logits = torch.gather(logits, dim = -1, index = sample_index)
        logits = logits.reshape(-1)

        # 若某个句子到达结束符, 分数保持不变
        score_eos = torch.zeros_like(logits)
        next_score = torch.where(flag == 1, score_eos, logits)
        sum_log = sum_log + next_score

        # 若某个句子到达结束符, 只需要添加结束标签
        sample_index = sample_index.reshape(-1)
        add_eos = torch.empty_like(sample_index).fill_(eos_index[0])
        sample_index = torch.where(flag == 1, add_eos, sample_index).reshape(-1, 1)
        caption_index = torch.cat([caption_index, sample_index], dim = 1)

    return caption_index


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_dim, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(512, config.hidden_dim)

        self.LayerNorm = torch.nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        input_shape = x.size()
        seq_length = input_shape[1]
        device = x.device

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand(input_shape)

        input_embeds = self.word_embeddings(x)
        position_embeds = self.position_embeddings(position_ids)

        embeddings = input_embeds + position_embeds
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ImgDecoder(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super(ImgDecoder, self).__init__()

        self.up_conv_relu = nn.Sequential(nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size = 2, stride = 2),
                                       nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1),
                                       nn.BatchNorm2d(out_channels),
                                       nn.ReLU())

    def forward(self, x):
        x = self.up_conv_relu(x)
        return x


class Generator(nn.Module):
    def __init__(self, config) -> None:
        super(Generator, self).__init__()

        self.head_nums = config.head_nums
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.mask_token_id = config.mask_token_id

        self.img_length = config.img_length

        self.img_mean = torch.tensor(config.image_mean)
        self.img_std = torch.tensor(config.image_std)

        temp_config = copy.deepcopy(config)
        temp_config.hidden_dim = config.generator_hidden_dim
        self.embeddings = Embeddings(temp_config)
        encoder_layer = nn.TransformerEncoderLayer(d_model = config.generator_hidden_dim, nhead = config.head_nums, 
                                                dropout = config.dropout, activation = F.relu, batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, config.generator_encoder_layer_nums)

        self.map = nn.Sequential(nn.Linear(2 * config.generator_hidden_dim, config.generator_hidden_dim),
                                 nn.LayerNorm(config.generator_hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(d_model = config.generator_hidden_dim, nhead = config.head_nums, 
                                                dropout = config.dropout, activation = F.relu, batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer, config.generator_decoder_layer_nums)

        self.upsample = nn.Sequential(ImgDecoder(config.generator_hidden_dim, 256),
                                      ImgDecoder(256, 128),
                                      ImgDecoder(128, 64),
                                      ImgDecoder(64, 32),
                                      ImgDecoder(32, 3),
                                      nn.Tanh())

    def forward(self, caption_index):

        caption_embedding = self.embeddings(caption_index)
        padding_mask = ((caption_index != self.pad_token_id) & (caption_index != self.eos_token_id)).to(torch.float32)
        caption_embedding = self.encoder(caption_embedding, src_key_padding_mask = ~(padding_mask > 0))

        global_caption_embedding = torch.sum(caption_embedding * padding_mask.unsqueeze(-1).repeat(1, 1, caption_embedding.size(-1)), dim = 1) / torch.sum(padding_mask, dim = 1, keepdim = True)

        noise = torch.randn((caption_embedding.size(0), self.img_length * self.img_length, caption_embedding.size(-1)), device = caption_embedding.device)
        img_embedding = torch.cat([global_caption_embedding.unsqueeze(1).repeat(1, noise.size(1), 1), noise], dim = -1)
        img_embedding = self.map(img_embedding)

        img_embedding = self.decoder(tgt = img_embedding, memory = caption_embedding, memory_key_padding_mask = ~(padding_mask > 0))
        img_embedding = img_embedding.permute(0, 2, 1).reshape(img_embedding.size(0), -1, self.img_length, self.img_length)
        img = self.upsample(img_embedding)

        img_mean = self.img_mean.to(img.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        img_std = self.img_std.to(img.device).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        img = (img - img_mean) / img_std

        return img

class ImgEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) -> None:
        super(ImgEncoder, self).__init__()

        self.down_conv_relu = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                       nn.BatchNorm2d(out_channels),
                                       nn.LeakyReLU(0.2))

    def forward(self, x):
        x = self.down_conv_relu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, config) -> None:
        super(Discriminator, self).__init__()

        self.image_encoder = nn.Sequential(ImgEncoder(3, 64, 7, 2, 3),
                                      ImgEncoder(64, 64, 5, 2, 2),
                                      ImgEncoder(64, 128, 3, 2, 1),
                                      ImgEncoder(128, 128, 3, 1, 1),
                                      ImgEncoder(128, 256, 3, 2, 1),
                                      ImgEncoder(256, config.discriminator_hidden_dim, 3, 1, 1))

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(config.discriminator_hidden_dim, 1)

    def forward(self, img):

        img = img + torch.randn_like(img)

        img_embedding = self.image_encoder(img)
        img_embedding = self.avg_pool(img_embedding)

        img_embedding = img_embedding.reshape(img_embedding.size(0), -1)
        score = self.linear(img_embedding).squeeze(-1)

        return score