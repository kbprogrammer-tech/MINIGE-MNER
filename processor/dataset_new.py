import random
import os
import torch
import json
import ast
import pdb
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from torchvision import transforms
import logging
logger = logging.getLogger(__name__)


class MMREProcessor(object):
    def __init__(self, data_path, bert_name):
        self.data_path = data_path
        self.re_path = data_path['re_path']
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
        self.tokenizer.add_special_tokens({'additional_special_tokens':['<s>', '</s>', '<o>', '</o>']})

    def load_from_file(self, mode="train", sample_ratio=1.0):
        """
        Args:
            mode (str, optional): dataset mode. Defaults to "train".
            sample_ratio (float, optional): sample ratio in low resouce. Defaults to 1.0.
        """        
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            words, relations, heads, tails, imgids, dataid = [], [], [], [], [], []
            for i, line in enumerate(lines):
                line = ast.literal_eval(line)   # str to dict
                words.append(line['token'])
                relations.append(line['relation'])
                heads.append(line['h']) # {name, pos}
                tails.append(line['t'])
                imgids.append(line['img_id'])
                dataid.append(i)

        assert len(words) == len(relations) == len(heads) == len(tails) == (len(imgids))

        # aux image
        aux_path = self.data_path[mode+"_auximgs"]
        aux_imgs = torch.load(aux_path)

         # sample
        if sample_ratio != 1.0:
            sample_indexes = random.choices(list(range(len(words))), k=int(len(words)*sample_ratio))
            sample_words = [words[idx] for idx in sample_indexes]
            sample_relations = [relations[idx] for idx in sample_indexes]
            sample_heads = [heads[idx] for idx in sample_indexes]
            sample_tails = [tails[idx] for idx in sample_indexes]
            sample_imgids = [imgids[idx] for idx in sample_indexes]
            sample_dataid = [dataid[idx] for idx in sample_indexes]
            assert len(sample_words) == len(sample_relations) == len(sample_imgids), "{}, {}, {}".format(len(sample_words), len(sample_relations), len(sample_imgids))
            return {'words':sample_words, 'relations':sample_relations, 'heads':sample_heads, 'tails':sample_tails, \
                 'imgids':sample_imgids, 'dataid': sample_dataid, 'aux_imgs':aux_imgs}
        
        return {'words':words, 'relations':relations, 'heads':heads, 'tails':tails, 'imgids': imgids, 'dataid': dataid, 'aux_imgs':aux_imgs}
       

    def get_relation_dict(self):
        with open(self.re_path, 'r', encoding="utf-8") as f:
            line = f.readlines()[0]
            re_dict = json.loads(line)
        return re_dict

class MMPNERProcessor(object):
    def __init__(self, data_path, bert_name) -> None:
        self.data_path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_name, do_lower_case=True)
    
    def load_from_file(self, mode="train", sample_ratio=1.0):
        """
        Args:
            mode (str, optional): dataset mode. Defaults to "train".
            sample_ratio (float, optional): sample ratio in low resouce. Defaults to 1.0.
        """   
        load_file = self.data_path[mode]
        logger.info("Loading data from {}".format(load_file))
        with open(load_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            raw_words, raw_targets = [], []
            raw_word, raw_target = [], []
            imgs = []
            imgs_dif = []
            # aux_imgs = []

            aux_path = self.data_path[mode + "_auximgs"]
            aux_imgs = torch.load(aux_path)
            aux_path_dif = self.data_path[mode + "_auximgs_dif"]
            aux_imgs_dif = torch.load(aux_path_dif)

            for line in lines:
                if line.startswith("IMGID:"):
                    img_id = line.strip().split('IMGID:')[1] + '.jpg'
                    # aux_imgs_ = []
                    # for etype in ['person', 'organization', 'location', 'miscellaneous']:
                    #     aux_imgs_.append('crop_{}_{}'.format(etype, img_id))

                    img_id_dif = line + '.png'
                    imgs.append(img_id)
                    imgs_dif.append(img_id_dif)
                    # aux_imgs.append(aux_imgs_)
                    continue
                if line != "\n":
                    raw_word.append(line.split('\t')[0])
                    label = line.split('\t')[1][:-1]
                    if 'OTHER' in label:
                        label = label[:2] + 'MISC'
                    raw_target.append(label)
                else:
                    raw_words.append(raw_word)
                    raw_targets.append(raw_target)
                    raw_word, raw_target = [], []

        assert len(raw_words) == len(raw_targets), "{}, {}".format(len(raw_words), len(raw_targets))
        # assert len(raw_words) == len(raw_targets) == len(imgs), "{}, {}, {}".format(len(raw_words), len(raw_targets),
        #                                                                             len(imgs))
        # sample data, only for low-resource
        if sample_ratio != 1.0:
            sample_indexes = random.choices(list(range(len(raw_words))), k=int(len(raw_words)*sample_ratio))
            sample_raw_words = [raw_words[idx] for idx in sample_indexes]
            sample_raw_targets = [raw_targets[idx] for idx in sample_indexes]
            sample_imgs = [imgs[idx] for idx in sample_indexes]
            imgs_dif = [imgs_dif[idx] for idx in sample_indexes]
            assert len(sample_raw_words) == len(sample_raw_targets) == len(sample_imgs), "{}, {}, {}".format(len(sample_raw_words), len(sample_raw_targets), len(sample_imgs))
            return {"words": sample_raw_words, "targets": sample_raw_targets, "imgs": sample_imgs, "aux_imgs":aux_imgs,'imgs_dif': imgs_dif, 'aux_imgs_dif': aux_imgs_dif}
        # pdb.set_trace()
        return {"words": raw_words, "targets": raw_targets, "imgs": imgs, "aux_imgs":aux_imgs, 'imgs_dif': imgs_dif, 'aux_imgs_dif': aux_imgs_dif}

    def get_label_mapping(self):
        LABEL_LIST = ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "X", "[CLS]", "[SEP]"]
        label_mapping = {label:idx for idx, label in enumerate(LABEL_LIST, 1)}
        label_mapping["PAD"] = 0
        return label_mapping

class MMREDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, max_seq=40, sample_ratio=1.0, mode="train") -> None:
        self.processor = processor
        self.transform = transform
        self.max_seq = max_seq
        self.img_path = img_path[mode]  if img_path is not None else img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.mode = mode

        self.data_dict = self.processor.load_from_file(mode, sample_ratio)
        self.re_dict = self.processor.get_relation_dict()
        self.tokenizer = self.processor.tokenizer

    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        word_list, relation, head_d, tail_d, imgid = self.data_dict['words'][idx], self.data_dict['relations'][idx], self.data_dict['heads'][idx], self.data_dict['tails'][idx], self.data_dict['imgids'][idx]
        item_id = self.data_dict['dataid'][idx]
        # [CLS] ... <s> head </s> ... <o> tail <o/> .. [SEP]
        head_pos, tail_pos = head_d['pos'], tail_d['pos']
        # insert <s> <s/> <o> <o/>
        extend_word_list = []
        for i in range(len(word_list)):
            if  i == head_pos[0]:
                extend_word_list.append('<s>')
            if i == head_pos[1]:
                extend_word_list.append('</s>')
            if i == tail_pos[0]:
                extend_word_list.append('<o>')
            if i == tail_pos[1]:
                extend_word_list.append('</o>')
            extend_word_list.append(word_list[i])
        extend_word_list = " ".join(extend_word_list)
        encode_dict = self.tokenizer.encode_plus(text=extend_word_list, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']
        input_ids, token_type_ids, attention_mask = torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask)
        
        re_label = self.re_dict[relation]   # label to id

         # image process
        if self.img_path is not None:
            try:
                img_path = os.path.join(self.img_path, imgid)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            if self.aux_img_path is not None:
                # process aux image
                aux_imgs = []
                aux_img_paths = []
                # pdb.set_trace()
                imgid = imgid.split(".")[0]
                if item_id in self.data_dict['aux_imgs']:
                    aux_img_paths  = self.data_dict['aux_imgs'][item_id]
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
                # discaed more than 3 aux image
                for i in range(min(3, len(aux_img_paths))):
                    aux_img = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_img = self.transform(aux_img)
                    aux_imgs.append(aux_img)

                # padding zero if less than 3
                for i in range(3-len(aux_img_paths)):
                    aux_imgs.append(torch.zeros((3, 224, 224)))

                aux_imgs = torch.stack(aux_imgs, dim=0)
                assert len(aux_imgs) == 3
            return input_ids, token_type_ids, attention_mask, torch.tensor(re_label), image, aux_imgs
        return input_ids, token_type_ids, attention_mask, torch.tensor(re_label)
    
class MMPNERDataset(Dataset):
    def __init__(self, processor, transform, img_path=None, aux_img_path=None, img_path_dif=None, max_seq=40, sample_ratio=1, mode='train', ignore_idx=0,dataset='twitter17') -> None:
        self.processor = processor
        self.transform = transform
        self.data_dict = processor.load_from_file(mode, sample_ratio)
        self.tokenizer = processor.tokenizer
        self.label_mapping = processor.get_label_mapping()
        self.max_seq = max_seq
        self.ignore_idx = ignore_idx
        self.img_path = img_path
        self.aux_img_path = aux_img_path[mode] if aux_img_path is not None else aux_img_path
        self.img_path_dif = img_path_dif
        self.mode = mode
        self.sample_ratio = sample_ratio
        self.dataset = dataset
    def __len__(self):
        return len(self.data_dict['words'])

    def __getitem__(self, idx):
        # word_list, label_list, img, img_dif, aux_imgs = self.data_dict['words'][idx], self.data_dict['targets'][idx], self.data_dict['imgs'][idx], self.data_dict['imgs_dif'][idx],self.data_dict['aux_imgs'][idx]
        word_list, label_list, img, img_dif = self.data_dict['words'][idx], self.data_dict['targets'][idx], \
        self.data_dict['imgs'][idx], self.data_dict['imgs_dif'][idx]
        tokens, labels = [], []
        for i, word in enumerate(word_list):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label = label_list[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(self.label_mapping[label])
                else:
                    labels.append(self.label_mapping["X"])
        if len(tokens) >= self.max_seq - 1:
            tokens = tokens[0:(self.max_seq - 2)]
            labels = labels[0:(self.max_seq - 2)]
        
        # pdb.set_trace()
        encode_dict = self.tokenizer.encode_plus(tokens, max_length=self.max_seq, truncation=True, padding='max_length')
        input_ids, token_type_ids, attention_mask = encode_dict['input_ids'], encode_dict['token_type_ids'], encode_dict['attention_mask']
        labels = [self.label_mapping["[CLS]"]] + labels + [self.label_mapping["[SEP]"]] + [self.ignore_idx]*(self.max_seq-len(labels)-2)
        
        # pdb.set_trace()

        if self.img_path is not None:
            # image process
            # pdb.set_trace()
            try:
                img_path = os.path.join(self.img_path, img)
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            except:
                img_path = os.path.join(self.img_path, 'inf.png')
                image = Image.open(img_path).convert('RGB')
                image = self.transform(image)
            # img_path_dif = os.path.join(self.img_path_dif, img_dif)
            if self.dataset =='twitter17':

                img_path_dif = os.path.join(self.img_path_dif, img_dif.replace(':', '_').replace('\n', ''))
            elif self.dataset =='twitter15':
                img_path_dif = os.path.join(self.img_path_dif, img_dif.replace(':', '_').replace('\n', '_'))
            image_dif1 = Image.open(img_path_dif).convert('RGB')
            # image_dif = self.processor.clip_processor(images=image_dif1, return_tensors='pt')['pixel_values'].squeeze()
            image_dif = self.transform(image_dif1)
            # if self.aux_img_path is not None:
            #     aux_imgs = []
            #     aux_img_paths = []
            #     if img in self.data_dict['aux_imgs']:
            #         aux_img_paths = self.data_dict['aux_imgs'][img]
            #         aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]
            #     for i in range(min(3, len(aux_img_paths))):
            #         aux_img = Image.open(aux_img_paths[i]).convert('RGB')
            #         aux_img = self.transform(aux_img)
            #         aux_imgs.append(aux_img)
            #
            #     for i in range(3-len(aux_img_paths)):
            #         aux_imgs.append(torch.zeros((3, 224, 224)))
            #
            #     aux_imgs = torch.stack(aux_imgs, dim=0)
            #     assert len(aux_imgs) == 3
            if self.aux_img_path is not None:
                aux_img_paths = []
                aux_images = []

                if img in self.data_dict['aux_imgs']:
                    aux_img_paths = self.data_dict['aux_imgs'][img]
                    # print(aux_img_paths)
                    aux_img_paths = [os.path.join(self.aux_img_path, path) for path in aux_img_paths]

                for i in range(min(3, len(aux_img_paths))):
                    aux_image = Image.open(aux_img_paths[i]).convert('RGB')
                    aux_image = self.transform(aux_image)
                    aux_images.append(aux_image)

                for i in range(3 - len(aux_images)):
                    aux_images.append(torch.zeros((3, 224, 224)))

                aux_images = torch.stack(aux_images, dim=0)
                assert len(aux_images) == 3
            if self.img_path_dif is not None:
                aux_imgs_dif = []
                for i in range(3):
                    # get the visual objects for generated images by coordinates
                    if len(self.data_dict['aux_imgs_dif'][img_dif.replace('\n', '')]) > i:
                        try:
                            xy = self.data_dict['aux_imgs_dif'][img_dif.replace('\n', '')][i].tolist()
                            aux_img_dif = image_dif1.crop((xy[0], xy[1], xy[2], xy[3]))
                            img_features_dif = self.transform(aux_img_dif)
                            aux_imgs_dif.append(img_features_dif)
                        except:
                            aux_imgs_dif.append(torch.zeros((3, 224, 224)))

                # visual objects padding
                for i in range(3 - len(aux_imgs_dif)):
                    aux_imgs_dif.append(torch.zeros((3, 224, 224)))
                aux_imgs_dif = torch.stack(aux_imgs_dif, dim=0)
                assert len(aux_imgs_dif) == 3

            if self.aux_img_path and self.img_path_dif:
                return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels), image, aux_images, image_dif, aux_imgs_dif
            elif self.aux_img_path:
                return torch.tensor(input_ids), torch.tensor(
                    attention_mask), torch.tensor(labels), image, aux_images
            elif self.img_path_dif:
                return torch.tensor(input_ids), torch.tensor(
                    attention_mask), torch.tensor(labels), image_dif, aux_imgs_dif
        assert len(input_ids) == len(token_type_ids) == len(attention_mask) == len(labels)
        return torch.tensor(input_ids), torch.tensor(token_type_ids), torch.tensor(attention_mask), torch.tensor(labels)
