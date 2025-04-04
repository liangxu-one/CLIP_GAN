import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPModel, GPT2LMHeadModel

class CaptionModel(nn.Module):
    def __init__(self, config) -> None:
        super(CaptionModel, self).__init__()

        self.head_nums = config.head_nums
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id
        self.mask_token_id = config.mask_token_id

        self.generation_length = config.generation_length
        self.img_length = config.img_length
        self.max_length = config.max_length
        self.sentence_nums = config.sentence_nums
        self.encoder_layer_nums = config.encoder_layer_nums
        self.decoder_layer_nums = config.decoder_layer_nums

        self.generator_fun = config.generator_fun
        self.vision_model = config.vision_model
        self.text_model = config.text_model

        if self.vision_model == 'swin':
            self.image_encoder = AutoModel.from_pretrained(config.vision_path)
            self.mlp = nn.Linear(self.image_encoder.num_features, self.hidden_dim)
        else:
            clip_model = CLIPModel.from_pretrained(config.vision_path)
            self.image_encoder = clip_model.vision_model
            self.mlp = nn.Linear(clip_model.vision_embed_dim, self.hidden_dim)

        self.img_to_text = nn.Parameter(torch.randn((config.gpt_prefix_length, self.hidden_dim)))
        map_layer = nn.TransformerDecoderLayer(d_model = config.hidden_dim, nhead = config.head_nums, 
                                            dropout = config.dropout, activation = F.relu, batch_first = True)
        self.map = nn.TransformerDecoder(map_layer, config.map_layer_nums)

        self.decoder_model = GPT2LMHeadModel.from_pretrained(config.text_path)
        self.decoder_model.resize_token_embeddings(len(config.tokenizer))

        if self.text_model == 'bert':
            self.loss_fun = nn.CrossEntropyLoss(reduction = 'none', ignore_index = config.pad_token_id)
        else:
            self.loss_fun = nn.CrossEntropyLoss(reduction = 'none')

    def forward(self, img, caption_index = None, caption_mask = None, label = None, is_label_data = None, img_embed = None):

        if img_embed is None:
            if self.vision_model == 'swin':
                img_embedding = self.image_encoder(img)[0]
            else:
                img_embedding = self.image_encoder(img)[0][:, 1:, ]
            img_embedding = self.mlp(img_embedding)
            img_to_text_feat = self.img_to_text.unsqueeze(0).repeat(img_embedding.size(0), 1, 1)
            img_embedding = self.map(tgt = img_to_text_feat, memory = img_embedding)
        else:
            img_embedding = img_embed

        if caption_index is None:
            return img_embedding

        if caption_index.dim() == 3:
            img_embedding = img_embedding.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, img_embedding.size(-2), img_embedding.size(-1))
            caption_mask = caption_mask.unsqueeze(1).repeat(1, caption_index.size(1), 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
            if is_label_data is not None:
                is_label_data = is_label_data.unsqueeze(1).repeat(1, caption_index.size(1), 1).reshape(-1, is_label_data.size(-1))
            caption_index = caption_index.reshape(-1, caption_index.size(-1))

        caption_mask = caption_mask.unsqueeze(1).repeat(1, self.head_nums, 1, 1).reshape(-1, caption_mask.size(-1), caption_mask.size(-1))
        padding_mask = ((caption_index != self.pad_token_id) & (caption_index != self.eos_token_id)).to(torch.float32)

        caption_embedding = self.decoder_model.get_input_embeddings()(caption_index)
        fuse_embedding = torch.cat([img_embedding, caption_embedding], dim = 1)
        fuse_mask = torch.cat([torch.ones((img_embedding.size(0), img_embedding.size(1)), device = padding_mask.device), padding_mask], dim = 1)
        out = self.decoder_model(inputs_embeds = fuse_embedding, attention_mask = (fuse_mask > 0))[0]
        pred = torch.split(out, [img_embedding.size(1), caption_embedding.size(1)], dim = 1)[1]

        if label is None:
            return pred
        else:
            pred = pred.reshape(-1, self.vocab_size)
            label = label.reshape(-1)
            loss = self.loss_fun(pred, label)
            if self.text_model == 'bert':
                loss = torch.sum(loss.reshape(-1, caption_index.size(-1)), dim = -1)
            else:
                loss = torch.sum(loss.reshape(-1, caption_index.size(-1)) * padding_mask, dim = -1)

            if is_label_data is not None:

                is_label_data = is_label_data.reshape(-1)
                data_with_label = torch.nonzero(is_label_data > 0)
                if data_with_label.size(0) > 0:
                    data_with_label = data_with_label.reshape(-1)
                    loss = torch.index_select(loss, dim = 0, index = data_with_label)
                    loss = torch.sum(loss) / data_with_label.size(0)
                else:
                    loss = 0 * torch.sum(loss)

            else:
                loss = torch.sum(loss) / caption_index.size(0)

            return loss