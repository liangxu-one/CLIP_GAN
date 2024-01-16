import os
import random
import torch
import copy
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_cosine_schedule_with_warmup, CLIPModel
from eval import eval
from config import Config
from model import CaptionModel
from read_file import build_data
from module.utils import Generator, Discriminator
from torchvision import transforms

def set_seed(seed):
    random.seed(seed)  # 配置Python random库的随机种子
    np.random.seed(seed)  # 配置Numpy库的随机种子
    torch.manual_seed(seed)  # 配置torch的随机种子
    torch.cuda.manual_seed(seed)  # 配置单个GPU上的种子
    torch.cuda.manual_seed_all(seed)  # 配置所有GPU上的种子
    # # cuDNN使用非确定性算法，并且可以使用torch.backends.cudnn.enabled = False来进行禁用
    # # 如果设置为torch.backends.cudnn.enabled =True，说明设置为使用使用非确定性算法
    # torch.backends.cudnn.enabled = False
    # # 将benchmark设置为False会让cudnn在有多种算法可选的情况下选择固定的一种
    # # 假如是True的话，cudnn会对多种算法进行测试，找到在你硬件上运行最快的那个算法，
    # # 然后再固定使用这个算法进行计算。
    # # 假如模型输入不会变化，比较规则，那设置成True可能会提高性能
    # # 假如模型输入会变化，那设置成True反而可能导致性能降低
    # # 不过要复现那还是设置成False吧~
    # torch.backends.cudnn.benchmark = False
    # # benchmark=False让选择的算法是固定的，然而这个算法本身可能还是non-deterministic的
    # # 所以设置deterministic=True可以让torch选择可确定的算法
    # torch.backends.cudnn.deterministic = True

def train(config):

    dist.init_process_group('nccl')
    # 分布式训练时获取该进程的rank值, 并只让rank为0的进程进行测试与指标评估
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device("cuda:{}".format(rank) if config.use_cuda else "cpu")

    # 加载模型, model为图片描述模型, Generator为文本生成图片, Discriminator为判别器
    model = CaptionModel(config)
    model.load_state_dict(torch.load(os.path.join(config.model_save_path, config.ck), map_location='cpu'))
    if rank == 0:
        print(model)
    model.to(device)
    if config.vision_model == 'swin':
        model = torch.nn.parallel.DistributedDataParallel(model, [device])
    else:
        model = torch.nn.parallel.DistributedDataParallel(model, [device], find_unused_parameters = True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    discriminator_model = Discriminator(config)
    if rank == 0:
        print(discriminator_model)
    discriminator_model.to(device)
    discriminator_model = torch.nn.parallel.DistributedDataParallel(discriminator_model, [device], broadcast_buffers = False)
    discriminator_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(discriminator_model)

    generator_model = Generator(config)
    if rank == 0:
        print(generator_model)
    generator_model.to(device)
    generator_model = torch.nn.parallel.DistributedDataParallel(generator_model, [device])
    generator_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(generator_model)

    clip_model = CLIPModel.from_pretrained(config.clip_path)
    if rank == 0:
        print(clip_model)
    clip_model.to(device)
    clip_model = torch.nn.parallel.DistributedDataParallel(clip_model, [device])
    clip_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(clip_model)
    for p in clip_model.module.parameters():
        p.requires_grad = False
    # 数据增强
    data_augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees = 15),  # 以15度范围内随机旋转
        transforms.ColorJitter(brightness = 0.2, contrast = 0.2),  # 随机调整亮度和对比度
    ])

    if rank == 0:
        # 读取数据
        print("读取数据")

    train_dict = build_data(config)
    train_sampler = DistributedSampler(train_dict, seed = config.seed)
    train_data = DataLoader(train_dict, config.batch_size, shuffle = (train_sampler is None), 
                            sampler = train_sampler, num_workers = config.num_workers)

    if rank == 0:
        configVal = Config(TrainOrVal = 'val')
        val_dict = build_data(configVal)
        val_data = DataLoader(val_dict, configVal.batch_size, shuffle = False, num_workers = configVal.num_workers)

        configTest = Config(TrainOrVal = 'test')
        test_dict = build_data(configTest)
        test_data = DataLoader(test_dict, configTest.batch_size, shuffle = False, num_workers = configTest.num_workers)

        print("train data is: ", len(train_dict))
        print("val data is: ", len(val_dict))
        print("test data is: ", len(test_dict))
        print("读取数据结束")

    pre_trained_params = list(map(id, model.module.image_encoder.parameters()))
    base_params = filter(lambda p: id(p) not in pre_trained_params, model.module.parameters())

    if config.vision_model == 'swin':
        optimizer = torch.optim.AdamW([
            {'params':model.module.image_encoder.parameters(), 'lr':config.lr/10},
            {'params':base_params, 'lr':config.lr},],
            lr=config.lr, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.AdamW([
            {'params':model.module.image_encoder.parameters(), 'lr':0},
            {'params':base_params, 'lr':config.lr},],
            lr=config.lr, weight_decay=config.weight_decay)
    # optimizer = torch.optim.Adam(model.module.parameters(), lr = config.lr, weight_decay = config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.epoch * len(train_data) // 10, 
                                                config.epoch * len(train_data))

    discriminator_optimizer = torch.optim.AdamW(discriminator_model.module.parameters(), lr = config.lr, weight_decay = config.weight_decay)
    discriminator_scheduler = get_cosine_schedule_with_warmup(discriminator_optimizer, config.epoch * len(train_data) // 10, 
                                                config.epoch * len(train_data))

    generator_optimizer = torch.optim.AdamW(generator_model.module.parameters(), lr = config.lr, weight_decay = config.weight_decay)
    generator_scheduler = get_cosine_schedule_with_warmup(generator_optimizer, config.epoch * len(train_data) // 10, 
                                                config.epoch * len(train_data))

    # 开始训练, 训练过程中, 前一半epoch只训练判别器与生成器, 后一半epoch三者同时开始训练
    for epoch in range(config.epoch):
        if rank == 0:
            print('R_lr: ', scheduler.get_last_lr())
            print('D_lr: ', discriminator_scheduler.get_last_lr())
            print('G_lr: ', generator_scheduler.get_last_lr())

        train_sampler.set_epoch(epoch)
        model.train()
        discriminator_model.train()
        generator_model.train()

        for i, batch in enumerate(train_data):
            for j in range(len(batch)):
                batch[j] = batch[j].to(device)

            model.zero_grad()
            discriminator_model.zero_grad()
            generator_model.zero_grad()

            # 训练判别器
            caption_index = copy.deepcopy(batch[1])
            with torch.no_grad():
                caption_index = caption_index.reshape(-1, caption_index.size(-1))
                fake_img = generator_model(caption_index)

            real_img = copy.deepcopy(batch[0])
            # fuse_img = torch.cat([real_img, fake_img], dim = 0)
            # score = discriminator_model(fuse_img)
            # real_score, fake_score = torch.split(score, [real_img.size(0), fake_img.size(0)], dim = 0)
            real_score = discriminator_model(real_img)
            fake_score = discriminator_model(fake_img)
            real_loss = torch.sum(real_score) / real_score.size(0)
            fake_loss = torch.sum(fake_score) / fake_score.size(0)
            dis_loss = -(real_loss - fake_loss)

            dis_loss.backward()
            discriminator_optimizer.step()
            discriminator_scheduler.step()

            # 判别器权重约束
            for p in discriminator_model.module.parameters():
                p.data.clamp_(-1, 1)

            model.zero_grad()
            discriminator_model.zero_grad()
            generator_model.zero_grad()

            # 训练生成器, 生成器loss由判别器与clip模型产生
            for p in discriminator_model.module.parameters():
                p.requires_grad = False
            for p in model.module.parameters():
                p.requires_grad = False

            caption_index = copy.deepcopy(batch[1])
            caption_index = caption_index.reshape(-1, caption_index.size(-1))
            fake_img = generator_model(caption_index)

            fake_batch = copy.deepcopy(batch[1:4])
            if batch[1].dim() == 3:
                fake_batch[0] = fake_batch[0].reshape(-1, fake_batch[0].size(-1))
                fake_batch[1] = fake_batch[1].unsqueeze(1).repeat(1, batch[1].size(1), 1, 1).reshape(-1, fake_batch[1].size(-2), fake_batch[1].size(-1))
                fake_batch[2] = fake_batch[2].reshape(-1, fake_batch[2].size(-1))

            fake_score = discriminator_model(fake_img)
            fake_loss = torch.sum(fake_score) / fake_score.size(0)
            aug_fake_img = data_augmentation(fake_img)
            clip_score = clip_model(caption_index, aug_fake_img, (caption_index > 0))
            clip_score = torch.diag(clip_score.logits_per_image)
            reconstructor_loss = torch.sum(clip_score) / clip_score.size(0)
            gen_loss = - fake_loss - reconstructor_loss

            gen_loss.backward()
            generator_optimizer.step()
            generator_scheduler.step()

            for p in discriminator_model.module.parameters():
                p.requires_grad = True
            for p in model.module.parameters():
                p.requires_grad = True

            # 当训练轮次超过一半时, 开始训练图像描述模型
            if epoch >= config.epoch // 2:
                model.zero_grad()
                discriminator_model.zero_grad()
                generator_model.zero_grad()

                # 训练图像描述模型, 未配对数据使用生成图片, 配对数据使用真实图片
                caption_index = copy.deepcopy(batch[1])
                with torch.no_grad():
                    caption_index = caption_index.reshape(-1, caption_index.size(-1))
                    fake_img = generator_model(caption_index)

                fake_batch = copy.deepcopy(batch[:4])
                fake_batch[0] = copy.deepcopy(fake_img)
                if batch[1].dim() == 3:
                    fake_batch[1] = fake_batch[1].reshape(-1, fake_batch[1].size(-1))
                    fake_batch[2] = fake_batch[2].unsqueeze(1).repeat(1, batch[1].size(1), 1, 1).reshape(-1, fake_batch[2].size(-2), fake_batch[2].size(-1))
                    fake_batch[3] = fake_batch[3].reshape(-1, fake_batch[3].size(-1))

                if len(batch) == 5:
                    real_img = copy.deepcopy(batch[0])
                    is_label_data = copy.deepcopy(batch[-1])
                    if batch[1].dim() == 3:
                        real_img = real_img.unsqueeze(1).repeat(1, batch[1].size(1), 1, 1, 1).reshape(-1, real_img.size(-3), real_img.size(-2), real_img.size(-1))
                        is_label_data = is_label_data.unsqueeze(1).repeat(1, batch[1].size(1), 1).reshape(-1, is_label_data.size(-1))

                    is_label_data = is_label_data.reshape(-1)
                    data_with_label = torch.nonzero(is_label_data > 0)
                    if data_with_label.size(0) > 0:
                        data_with_label = data_with_label.reshape(-1)
                        fake_batch[0][data_with_label] = real_img[data_with_label]

                re_loss = model(*fake_batch)

                re_loss.backward()
                optimizer.step()
                scheduler.step()

            if rank == 0 and i % 100 == 0:
                if epoch >= config.epoch // 2:
                    print('i/batch: {}/{} | epoch/epochs: {}/{} | R_loss: {}, D_loss: {}, G_loss: {}, G_D_loss: {}, G_R_loss: {}'.format(i, len(train_data), epoch, config.epoch, re_loss.item(), dis_loss.item(), gen_loss.item(), fake_loss.item(), reconstructor_loss.item()))
                else:
                    print('i/batch: {}/{} | epoch/epochs: {}/{} | R_loss: {}, D_loss: {}, G_loss: {}, G_D_loss: {}, G_R_loss: {}'.format(i, len(train_data), epoch, config.epoch, 0, dis_loss.item(), gen_loss.item(), fake_loss.item(), reconstructor_loss.item()))

                # # 生成器保存一张照片, 需要将图片反归一化
                # fake_img_sample = fake_img[0].data.cpu()
                # img_mean = torch.tensor(config.image_mean).unsqueeze(1).unsqueeze(2)
                # img_std = torch.tensor(config.image_std).unsqueeze(1).unsqueeze(2)
                # fake_img_sample = fake_img_sample * img_std + img_mean
                # fake_img_sample = torch.clamp(fake_img_sample, 0, 1)
                # fake_img_sample = config.tensor_to_PILImage(fake_img_sample)
                # fake_img_sample.save(config.img_save_path + 'fake_img_{}_{}.jpg'.format(epoch, i//100))

        if rank == 0:
            torch.save(model.module.state_dict(), os.path.join(config.model_save_path, 'gan_epoch_{}.pt'.format(epoch)))
            print("test:", end = ' ')
            with torch.no_grad():
                eval(configVal, model.module, val_data, val_dict)

    if rank == 0:
        with torch.no_grad():
            eval(configTest, model.module, test_data, test_dict)

if __name__ == '__main__':
    set_seed(Config().seed)
    config = Config(stage = 'gan')
    train(config)