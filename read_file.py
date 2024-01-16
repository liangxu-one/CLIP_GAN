import json
import torch
import random
from PIL import Image
from torch.utils.data import Dataset

class ImageCaption(Dataset):
    def __init__(self, config, data_labeled) -> None:
        super(ImageCaption, self).__init__()

        self.with_rl = config.with_rl

        self.TrainOrVal = config.TrainOrVal
        self.max_length = config.max_length
        self.sentence_nums = config.sentence_nums

        self.img_dir = config.image_dir

        # 从图片名找到图片id
        self.filename_to_id = {}
        # 从图片id找到句子
        self.imgid_to_sentences = {}

        info_json = open(config.info_json, encoding = 'utf-8')
        content = json.load(info_json)
        for item in content['images']:
            file_name = item['filename']
            img_id = int(item['imgid'])
            self.filename_to_id[file_name] = img_id

            self.imgid_to_sentences[img_id] = []
            for sentence in item['sentences']:
                tokens = ' '.join(sentence['tokens'])
                self.imgid_to_sentences[img_id].append(tokens)

        img_name_file = open(config.image_name, encoding = 'utf-8')
        img_name = img_name_file.readlines()

        if self.TrainOrVal == 'train':
            is_label_data = open(config.is_label_data, encoding = 'utf-8')
            self.image_is_has_caption = {}
            for item in is_label_data.readlines():
                file_name = str(item.split(' ')[0])
                is_label = int(item.split(' ')[1].split('\n')[0])
                self.image_is_has_caption[file_name] = is_label

        # 打乱未配对图片
        if self.TrainOrVal == 'train':
            unpair_image_id = []
            unpair_caption = []
            for file_name in img_name:
                file_name = file_name.split('\n')[0]
                if self.image_is_has_caption[file_name] == data_labeled:
                    continue
                else:
                    image_id = self.filename_to_id[file_name]
                    unpair_image_id.append(image_id)
                    unpair_caption.append(self.imgid_to_sentences[image_id])

            # 取一定数量的不匹配图文
            unpair_image_id = unpair_image_id[:int(len(unpair_image_id) * config.unlabeled_data_proportion)]
            unpair_caption = unpair_caption[:int(len(unpair_caption) * config.unlabeled_data_proportion)]
            assert len(unpair_image_id) == len(unpair_caption)

            print("未配对图文对: ", len(unpair_image_id))
            random.shuffle(unpair_image_id)
            random.shuffle(unpair_caption)
            unpair_imgid_to_sentences = dict(zip(unpair_image_id, unpair_caption))
            for image_id in unpair_imgid_to_sentences.keys():
                self.imgid_to_sentences[image_id] = unpair_imgid_to_sentences[image_id]

        self.dataset = []

        # # 将一张图片和每个描述当作一个样本
        # if self.TrainOrVal == 'train':
        #     for file_name in img_name:

        #         file_name = file_name.split('\n')[0]

        #         if config.stage == 'supervised' and self.image_is_has_caption[file_name] != data_labeled:
        #             continue

        #         img_id = self.filename_to_id[file_name]
        #         for sentence in self.imgid_to_sentences[img_id]:
        #             temp = {}
        #             temp['img_id'] = img_id
        #             temp['file_name'] = file_name
        #             temp['sentence'] = sentence
        #             self.dataset.append(temp)
        # else:
        #     for file_name in img_name:
        #         file_name = file_name.split('\n')[0]
        #         temp = {}
        #         temp['img_id'] = self.filename_to_id[file_name]
        #         temp['file_name'] = file_name
        #         self.dataset.append(temp)

        # 将一张图片和多个对应描述当作一个样本
        for file_name in img_name:
            file_name = file_name.split('\n')[0]

            if config.stage == 'supervised' and self.TrainOrVal == 'train' and self.image_is_has_caption[file_name] != data_labeled:
                continue

            if config.stage == 'gan' and self.TrainOrVal == 'train' and self.image_is_has_caption[file_name] != data_labeled and self.filename_to_id[file_name] not in unpair_imgid_to_sentences:
                continue

            temp = {}
            temp['img_id'] = self.filename_to_id[file_name]
            temp['file_name'] = file_name
            self.dataset.append(temp)

        self.transforms = config.image_process

        self.tokenizer = config.tokenizer
        self.pad_token_id = config.pad_token_id
        self.bos_token_id = config.bos_token_id
        self.eos_token_id = config.eos_token_id

        self.mask = config.generator_fun(self.max_length)

    def __getitem__(self, index):

        item = self.dataset[index]
        image_path = self.img_dir + item['file_name']
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        image = self.transforms(image, return_tensors = 'pt')['pixel_values'].squeeze(0)

        if self.TrainOrVal == 'train' and self.with_rl == False:

            # # 将一张图片和每个描述当作一个样本
            # caption_sentence = item['sentence']
            # caption_encoded = self.tokenizer.encode_plus(
            # caption_sentence, max_length=self.max_length, padding=False, return_attention_mask=False, return_token_type_ids=False, truncation=True)

            # caption_token_id = caption_encoded['input_ids'][1:-1]
            # caption = [self.bos_token_id] + caption_token_id + [self.pad_token_id] * (self.max_length - 1 - len(caption_token_id))
            # label = caption_token_id + [self.eos_token_id] + [self.pad_token_id] * (self.max_length  - 1 - len(caption_token_id))

            # assert len(caption) == self.max_length and len(label) == self.max_length

            # caption = torch.tensor(caption)
            # label = torch.tensor(label)
            # is_label_data = torch.tensor(self.image_is_has_caption[item['file_name']])

            # 将一张图片和多个对应描述当作一个样本
            img_id = item['img_id']
            sentences = self.imgid_to_sentences[img_id][:self.sentence_nums]

            all_caption = []
            all_label = []

            for caption_sentence in sentences:
                caption_encoded = self.tokenizer.encode_plus(
                caption_sentence, max_length=self.max_length, padding=False, return_attention_mask=False, return_token_type_ids=False, truncation=True)

                caption_token_id = caption_encoded['input_ids'][1:-1]
                caption = [self.bos_token_id] + caption_token_id + [self.pad_token_id] * (self.max_length - 1 - len(caption_token_id))
                label = caption_token_id + [self.eos_token_id] + [self.pad_token_id] * (self.max_length  - 1 - len(caption_token_id))

                assert len(caption) == self.max_length and len(label) == self.max_length

                all_caption.append(caption)
                all_label.append(label)

            caption = torch.tensor(all_caption)
            label = torch.tensor(all_label)
            is_label_data = torch.tensor([self.image_is_has_caption[item['file_name']]])

            return image, caption, self.mask, label, is_label_data

        else:
            caption_index = torch.tensor([self.bos_token_id])

            return image, caption_index, item['img_id'], item['file_name']

    def __len__(self):
        return len(self.dataset)
        # return 1111 if self.TrainOrVal == 'train' else 1111

def build_data(config, data_labeled = 1):
    data = ImageCaption(config, data_labeled)
    return data