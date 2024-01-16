import os, sys
import torch.nn as nn
from transformers import AutoTokenizer, ViTImageProcessor, CLIPImageProcessor
import torchvision.transforms as transforms

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
path = sys.path[0]

class Config():
    def __init__(self, TrainOrVal = 'train', with_rl = False, stage = 'supervised') -> None:
        self.use_cuda = True
        self.gpu = 0
        self.num_workers = 4

        self.with_rl = with_rl

        self.seed = 0
        self.lr = 1e-4
        self.rl_lr = 1e-5
        self.weight_decay = 1e-4
        self.epoch = 30 if stage == 'supervised' else 20

        self.decode_method = 'greedy'
        self.beam_size = 5
        self.generator_fun = nn.Transformer().generate_square_subsequent_mask

        self.PreTrainedPath = ['PreTrainedModel/albert-base-v2',
                               'PreTrainedModel/bert-base-uncased',
                               'PreTrainedModel/swin-tiny-patch4-window7-224',
                               'PreTrainedModel/swin-base-patch4-window7-224-in22k',
                               'PreTrainedModel/swin-large-patch4-window12-384-in22k',
                               'PreTrainedModel/clip-vit-base-patch32']

        self.text_model = 'clip'
        self.text_path = os.path.join(path, self.PreTrainedPath[5])
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_path, do_lower_case = True)

        self.vision_model = 'swin'
        self.vision_path = os.path.join(path, self.PreTrainedPath[2])
        if self.vision_model == 'swin':
            self.image_process = ViTImageProcessor.from_pretrained(self.vision_path)
        else:
            self.image_process = CLIPImageProcessor.from_pretrained(self.vision_path)

        self.image_mean = self.image_process.image_mean
        self.image_std = self.image_process.image_std

        self.vocab_size = self.tokenizer.vocab_size

        if self.text_model == 'bert':
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token_id = self.tokenizer.cls_token_id
            self.eos_token_id = self.tokenizer.sep_token_id
            self.mask_token_id = self.tokenizer.mask_token_id

        else:
            self.pad_token_id = self.tokenizer.pad_token_id
            self.bos_token_id = self.tokenizer.bos_token_id
            self.eos_token_id = self.tokenizer.eos_token_id
            self.mask_token_id = self.tokenizer.mask_token_id

        self.batch_size = 24
        self.img_length = 7
        self.max_length = 25
        self.generation_length = self.max_length - 1
        self.dropout = 0.1
        self.head_nums = 8
        self.encoder_layer_nums = 2
        self.decoder_layer_nums = 1 if self.vision_model ==  'swin' else 4
        self.hidden_dim = 768

        self.sentence_nums = 5

        self.dataset = 'coco'
        self.TrainOrVal = TrainOrVal
        self.info_json = os.path.join(path, 'data/{}/dataset_{}.json'.format(self.dataset, self.dataset))
        self.image_dir = os.path.join(path, 'data/{}/img/'.format(self.dataset))
        self.image_name = os.path.join(path, 'data/{}/{}_{}.txt'.format(self.dataset, self.dataset, self.TrainOrVal))
        self.is_label_data = os.path.join(path, 'data/{}/is_label_data.txt'.format(self.dataset))
        self.unlabeled_data_proportion = 1

        self.stage = stage

        self.model_save_path = os.path.join(path, 'model_save/train_{}'.format(self.dataset))
        self.ck = 'baseline_tiny_clip_text.pt'

        # 判别器
        self.discriminator_hidden_dim = 512
        self.clip_path = os.path.join(path, self.PreTrainedPath[5])

        # 生成器
        self.generator_hidden_dim = 512
        self.generator_encoder_layer_nums = 2
        self.generator_decoder_layer_nums = 2
        self.img_save_path = os.path.join(path, 'data/fake_img/')
        self.tensor_to_PILImage = transforms.ToPILImage()