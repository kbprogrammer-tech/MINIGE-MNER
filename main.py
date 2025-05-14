import os
import pdb
import argparse
import logging
import sys
from os.path import join, dirname

sys.path.append("..")

import torch
import numpy as np
import random
from torchvision import transforms
from torch.utils.data import DataLoader
from models.model import HVPNeTREModel, HVPNeTNERModel
from processor.dataset_new import MMREProcessor, MMPNERProcessor, MMREDataset, MMPNERDataset
from modules.train import RETrainer, NERTrainer

import warnings
import time  # 新增：用于记录时间

warnings.filterwarnings("ignore", category=UserWarning)


def obtain_logger(filename):
    home = dirname(__file__)
    logger = logging.getLogger('statisticNew')
    logger.setLevel(logging.INFO)
    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(filename)
    fh.setLevel(logging.DEBUG)
    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # 给logger添加handler
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# 新增：计算模型参数量的函数
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


MODEL_CLASSES = {
    'MRE': HVPNeTREModel,
    'twitter15': HVPNeTNERModel,
    'twitter17': HVPNeTNERModel
}

TRAINER_CLASSES = {
    'MRE': RETrainer,
    'twitter15': NERTrainer,
    'twitter17': NERTrainer
}
DATA_PROCESS = {
    'MRE': (MMREProcessor, MMREDataset),
    'twitter15': (MMPNERProcessor, MMPNERDataset),
    'twitter17': (MMPNERProcessor, MMPNERDataset)
}

DATA_PATH = {
    'MRE': {
        # text data
        'train': 'D:/code/TMR-main/TMR-main/TMR-RE/data/txt/ours_train.txt',
        'dev': 'D:/code/TMR-main/TMR-main/TMR-RE/data/txt/ours_val.txt',
        'test': 'D:/code/TMR-main/TMR-main/TMR-RE/data/txt/ours_test.txt',
        # {data_id : object_crop_img_path}
        'train_auximgs': 'D:/code/TMR-main/TMR-main/TMR-RE/data/txt/mre_train_dict.pth',
        'dev_auximgs': 'D:/code/TMR-main/TMR-main/TMR-RE/data/txt/mre_val_dict.pth',
        'test_auximgs': 'D:/code/TMR-main/TMR-main/TMR-RE/data/txt/mre_test_dict.pth',
        # relation json data
        're_path': 'D:/code/TMR-main/TMR-main/TMR-RE/data/ours_rel2id.json',
        # visual objects data
        'train_auximgs_dif': 'D:/code/TMR-main/TMR-main/TMR-RE/data/txt/mre_dif_train_dict.pth',
        'dev_auximgs_dif': 'D:/code/TMR-main/TMR-main/TMR-RE/data/txt/mre_dif_val_dict.pth',
        'test_auximgs_dif': 'D:/code/TMR-main/TMR-main/TMR-RE/data/txt/mre_dif_test_dict.pth',
    },

    'twitter15': {
        ## input text data
        'train': 'D:/code/data/NER_data/twitter2015/train.txt',
        'dev': 'D:/code/data/NER_data/twitter2015/val.txt',
        'test': 'D:/code/data/NER_data/twitter2015/test.txt',

        # visual objects data
        'train_auximgs': 'D:/code/data/NER_data/twitter2015/twitter2015_train_dict.pth',
        'dev_auximgs': 'D:/code/data/NER_data/twitter2015/twitter2015_val_dict.pth',
        'test_auximgs': 'D:/code/data/NER_data/twitter2015/twitter2015_test_dict.pth',
        'train_auximgs_dif': 'D:/code/data/NER_data/twitter2015/train_grounding_cut_dif.pth',
        'dev_auximgs_dif': 'D:/code/data/NER_data/twitter2015/val_grounding_cut_dif.pth',
        'test_auximgs_dif': 'D:/code/data/NER_data/twitter2015/test_grounding_cut_dif.pth',

        # correlation coefficient data
        'train_weak_ori': 'D:/code/data/NER_data/twitter2015/ner_train_weight_weak.txt',
        'dev_weak_ori': 'D:/code/data/NER_data/twitter2015/ner_val_weight_weak.txt',
        'test_weak_ori': 'D:/code/data/NER_data/twitter2015/ner_test_weight_weak.txt',
        'train_strong_ori': 'D:/code/data/NER_data/twitter2015/ner_train_weight_strong.txt',
        'dev_strong_ori': 'D:/code/data/NER_data/twitter2015/ner_val_weight_strong.txt',
        'test_strong_ori': 'D:/code/data/NER_data/twitter2015/ner_test_weight_strong.txt',
        'train_weak_dif': 'D:/code/data/NER_data/twitter2015/ner_diff_train_weight_weak.txt',
        'dev_weak_dif': 'D:/code/data/NER_data/twitter2015/ner_diff_val_weight_weak.txt',
        'test_weak_dif': 'D:/code/data/NER_data/twitter2015/ner_diff_test_weight_weak.txt',
        'train_strong_dif': 'D:/code/data/NER_data/twitter2015/ner_diff_train_weight_strong.txt',
        'dev_strong_dif': 'D:/code/data/NER_data/twitter2015/ner_diff_val_weight_strong.txt',
        'test_strong_dif': 'D:/code/data/NER_data/twitter2015/ner_diff_test_weight_strong.txt',

        # phrase text data
        'train_grounding_text': 'D:/code/data/NER_data/twitter2015/ner15_grounding_text_train.json',
        'dev_grounding_text': 'D:/code/data/NER_data/twitter2015/ner15_grounding_text_val.json',
        'test_grounding_text': 'D:/code/data/NER_data/twitter2015/ner15_grounding_text_test.json',
    },

    'twitter17': {
        # text data
        'train': 'D:/code/data/NER_data/twitter2017/train.txt',
        'dev': 'D:/code/data/NER_data/twitter2017/valid.txt',
        'test': 'D:/code/data/NER_data/twitter2017/test.txt',
        # visual objects data
        'train_auximgs': 'D:/code/data/NER_data/twitter2017/twitter2017_train_dict.pth',
        # {data_id : object_crop_img_path}
        'dev_auximgs': 'D:/code/data/NER_data/twitter2017/twitter2017_val_dict.pth',
        'test_auximgs': 'D:/code/data/NER_data/twitter2017/twitter2017_test_dict.pth',
        'train_auximgs_dif': 'D:/code/data/NER_data/twitter2017/train_grounding_cut_dif.pth',
        'dev_auximgs_dif': 'D:/code/data/NER_data/twitter2017/val_grounding_cut_dif.pth',
        'test_auximgs_dif': 'D:/code/data/NER_data/twitter2017/test_grounding_cut_dif.pth'
    },

}

# image data
IMG_PATH = {
    'MRE': {'train': 'D:/code/TMR-main/TMR-main/TMR-RE/data/img_org/train/',
            'dev': 'D:/code/TMR-main/TMR-main/TMR-RE/data/img_org/val/',
            'test': 'D:/code/TMR-main/TMR-main/TMR-RE/data/img_org/test'},
    'twitter15': 'D:/code/data/NER_data/twitter2015_images',
    'twitter17': 'D:/code/data/NER_data/twitter2017_images',
}

# generated image data
IMG_PATH_dif = {
    'MRE': {
        'train': './data/re_diffusion_pic/train/',
        'test': './data/re_diffusion_pic/test/',
        'dev': './data/re_diffusion_pic/val/',
    },
    'twitter15': {
        'train': 'D:/code/data/NER_data/ner15_diffusion_pic/train/',
        'test': 'D:/code/data/NER_data/ner15_diffusion_pic/test/',
        'dev': 'D:/code/data/NER_data/ner15_diffusion_pic/val/',
    },
    'twitter17': {
        'train': 'D:/code/data/NER_data/twitter17_diff_images/',
        'test': 'D:/code/data/NER_data/twitter17_diff_images/',
        'dev': 'D:/code/data/NER_data/twitter17_diff_images/',
    }
}
# auxiliary images
AUX_PATH = {
    'MRE': {
        'train': 'D:/code/TMR-main/TMR-main/TMR-RE/data/img_vg/train/crops',
        'dev': 'D:/code/TMR-main/TMR-main/TMR-RE/data/img_vg/val/crops',
        'test': 'D:/code/TMR-main/TMR-main/TMR-RE/data/img_vg/test/crops'
    },
    'twitter15': {
        'train': 'D:/code/data/NER_data/twitter2015_aux_images/train/crops',
        'dev': 'D:/code/data/NER_data/twitter2015_aux_images/val/crops',
        'test': 'D:/code/data/NER_data/twitter2015_aux_images/test/crops',
    },

    'twitter17': {
        'train': 'D:/code/data/NER_data/twitter2017_aux_images/train/crops',
        'dev': 'D:/code/data/NER_data/twitter2017_aux_images/val/crops',
        'test': 'D:/code/data/NER_data/twitter2017_aux_images/test/crops'
    }
    # 'twitter15': '/data/NER_data/twitter2015_crop_images',
    #
    # 'twitter17': '/data/NER_data/twitter2017_crop_images'
}


def set_seed(seed=2021):
    """set random seed"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='twitter15', type=str, help="The name of dataset.")
    parser.add_argument('--bert_name', default='D:/code/bert-base-uncased', type=str,
                        help="Pretrained language model path")
    parser.add_argument('--num_epochs', default=30, type=int, help="num training epochs")
    parser.add_argument('--device', default='cuda', type=str, help="cuda or cpu")
    parser.add_argument('--batch_size', default=32, type=int, help="batch size")
    parser.add_argument('--bert_lr', default=5e-5, type=float, help="learning rate")
    parser.add_argument('--lr', default=5e-2, type=float, help="learning rate")
    parser.add_argument('--warmup_ratio', default=0.01, type=float)
    parser.add_argument('--eval_begin_epoch', default=3, type=int, help="epoch to start evluate")
    parser.add_argument('--seed', default=49, type=int, help="random seed, default is 1")
    parser.add_argument('--prompt_len', default=10, type=int, help="prompt length")
    parser.add_argument('--prompt_dim', default=800, type=int, help="mid dimension of prompt project layer")
    parser.add_argument('--output_dir', default='./data/output_dir', type=str)
    parser.add_argument('--notes', default="", type=str, help="input some remarks for making save path dir.")
    parser.add_argument('--use_prompt', default='True', action='store_true')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--max_seq', default=128, type=int)
    parser.add_argument('--ignore_idx', default=-100, type=int)
    parser.add_argument('--sample_ratio', default=1.0, type=float, help="only for low resource.")
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--log_name', default='test', type=str)
    parser.add_argument('--model_name', default='test', type=str)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--use_bias', action='store_true')
    parser.add_argument('--m', default=3, type=int)
    parser.add_argument('--beta', default=0.1, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.1, type=float)
    parser.add_argument('--beta3', default=0.1, type=float)
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--fusion', default='gate', type=str)
    parser.add_argument('--reduction', default='mean', type=str)
    parser.add_argument('--neg_num', default=1, type=int)
    parser.add_argument('--tune_resnet', action='store_true')
    parser.add_argument('--score_func', default='concat', type=str)
    parser.add_argument('--version', default='new', type=str)
    parser.add_argument('--crf_lr', default='0.05', type=float)
    parser.add_argument('--other_lr', default=1e-3, type=float)
    parser.add_argument('--max_norm', default=0, type=float)
    args = parser.parse_args()

    data_path, img_path, aux_path, img_path_dif = DATA_PATH[args.dataset_name], IMG_PATH[args.dataset_name], AUX_PATH[
        args.dataset_name], IMG_PATH_dif[args.dataset_name]
    model_class, Trainer = MODEL_CLASSES[args.dataset_name], TRAINER_CLASSES[args.dataset_name]
    data_process, dataset_class = DATA_PROCESS[args.dataset_name]

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    set_seed(args.seed)  # set seed, default is 1

    log_dir = os.path.join(os.path.join(args.output_dir, args.model_name), args.log_name)

    args.output_dir = os.path.join(log_dir, args.dataset_name)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if args.do_train and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    print(os.path.join(log_dir, args.dataset_name + ".log"))
    logger = obtain_logger(os.path.join(log_dir, args.dataset_name + ".log"))

    writer = None
    processor = data_process(data_path, args.bert_name)
    train_dataset = dataset_class(processor, transform, img_path, aux_path, img_path_dif['train'], args.max_seq,
                                  sample_ratio=args.sample_ratio, mode='train', ignore_idx=args.ignore_idx,
                                  dataset=args.dataset_name)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)

    dev_dataset = dataset_class(processor, transform, img_path, aux_path, img_path_dif['dev'], args.max_seq, mode='dev',
                                ignore_idx=args.ignore_idx, dataset=args.dataset_name)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                pin_memory=True)

    test_dataset = dataset_class(processor, transform, img_path, aux_path, img_path_dif['test'], args.max_seq,
                                 mode='test', ignore_idx=args.ignore_idx, dataset=args.dataset_name)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                 pin_memory=True)

    if args.dataset_name == 'MRE':  # RE task
        re_dict = processor.get_relation_dict()
        num_labels = len(re_dict)
        tokenizer = processor.tokenizer
        model = HVPNeTREModel(num_labels, tokenizer, args=args)

        trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                          processor=processor, args=args, logger=logger, writer=writer)
    else:  # NER task
        label_mapping = processor.get_label_mapping()
        label_list = list(label_mapping.keys())
        model = HVPNeTNERModel(label_list, args)

        trainer = Trainer(train_data=train_dataloader, dev_data=dev_dataloader, test_data=test_dataloader, model=model,
                          processor=processor, label_map=label_mapping, args=args, logger=logger, writer=writer)

    # 计算模型参数量
    num_params = count_parameters(model)
    logger.info(f"Number of trainable parameters: {num_params}")

    if args.do_train:
        logger.info("\nArgs: ")
        for arg in vars(args):
            logger.info("{}: {}".format(arg, getattr(args, arg)))
        logger.info("\n")

        # 记录训练开始时间
        start_time = time.time()
        # train
        trainer.train()
        # 记录训练结束时间
        end_time = time.time()
        # 计算训练时间
        training_time = end_time - start_time
        logger.info(f"Training time: {training_time:.2f} seconds")
        trainer.test()

    if args.only_test:
        # only do test
        trainer.test()

    torch.cuda.empty_cache()
    # writer.close()


if __name__ == "__main__":
    main()