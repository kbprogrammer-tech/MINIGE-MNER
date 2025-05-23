import os
import pdb
import torch
from torch import optim
from tqdm import tqdm
import numpy as np
import random
from sklearn.metrics import classification_report as sk_classification_report
from seqeval.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup

from .metrics import eval_result

class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()

class RETrainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, args=None, logger=None,  writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.re_dict = processor.get_relation_dict()
        self.logger = logger
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args
        self.before_multimodal_train()

    def train(self):
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data)*self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        # if self.args.load_path is not None:  # load model from load_path
        #     self.logger.info("Loading model from {}".format(self.args.load_path))
        #     self.model.load_state_dict(torch.load(self.args.load_path))
        #     self.logger.info("Load model successful!")
        
        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs+1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (loss, logits), labels = self._step(batch, mode="train")
                    avg_loss += loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
                        avg_loss = 0

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)   # generator to dev.
            
            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch, self.best_dev_metric))
            self.logger.info("Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch, self.best_test_metric))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        step = 0
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    (loss, logits), labels = self._step(batch, mode="dev")    # logits: batch, 3
                    total_loss += loss.detach().cpu().item()
                    
                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels, labels=list(self.re_dict.values())[1:], target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc']*100, 4), round(result['micro_f1']*100, 4)
                if self.writer:
                    self.writer.add_scalar(tag='dev_acc', scalar_value=acc, global_step=epoch)    # tensorbordx
                    self.writer.add_scalar(tag='dev_f1', scalar_value=micro_f1, global_step=epoch)    # tensorbordx
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss/len(self.test_data), global_step=epoch)    # tensorbordx

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}."\
                            .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch, micro_f1, acc))
                if micro_f1 >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = micro_f1 # update best metric(f1 score)
                    torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, "best_model.pth"))
                    self.logger.info("Save best model at {}".format(self.args.output_dir))
               

        self.model.train()

    def test(self):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        
        self.logger.info("Loading model from {}".format(self.args.output_dir))
        self.model.load_state_dict(torch.load( os.path.join(self.args.output_dir, "best_model.pth")))
        self.logger.info("Load model successful!")
        true_labels, pred_labels = [], []
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device  
                    (loss, logits), labels = self._step(batch, mode="dev")    # logits: batch, 3
                    total_loss += loss.detach().cpu().item()
                    
                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    
                    pbar.update()
                # evaluate done
                pbar.close()
                sk_result = sk_classification_report(y_true=true_labels, y_pred=pred_labels, labels=list(self.re_dict.values())[1:], target_names=list(self.re_dict.keys())[1:], digits=4)
                self.logger.info("%s\n", sk_result)
                result = eval_result(true_labels, pred_labels, self.re_dict, self.logger)
                acc, micro_f1 = round(result['acc']*100, 4), round(result['micro_f1']*100, 4)
                if self.writer:
                    self.writer.add_scalar(tag='test_acc', scalar_value=acc)    # tensorbordx
                    self.writer.add_scalar(tag='test_f1', scalar_value=micro_f1)    # tensorbordx
                    self.writer.add_scalar(tag='test_loss', scalar_value=total_loss/len(self.test_data))    # tensorbordx
                total_loss = 0
                self.logger.info("Test f1 score: {}, acc: {}.".format(micro_f1, acc))
                    
        self.model.train()
        
    def _step(self, batch, mode="train"):
        input_ids, token_type_ids, attention_mask, labels, images, aux_imgs = batch

        re_loss, dis_loss, txt_kdl, img_kdl, logits = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, images=images, aux_imgs=aux_imgs)
        loss = re_loss + self.args.beta1 * dis_loss + self.args.beta2 * txt_kdl + self.args.beta3 * img_kdl
        # pdb.set_trace()
        if mode == 'train' and self.step % 100 == 0:
            self.logger.info("re_loss: {}, dis_loss: {}, txt_kdl: {}, img_kdl: {}".format(re_loss.item(), dis_loss.item(), txt_kdl.item(), img_kdl.item()))
        return (loss, logits), labels

    def before_multimodal_train(self):


        if not self.args.tune_resnet:
            for name, par in self.model.named_parameters(): # freeze resnet
                if 'image_model' in name:   par.requires_grad = False

        parameters_to_optimize = []
        params1 = {'lr':self.args.lr, 'weight_decay':1e-2, 'params': []}
        params2 = {'lr':self.args.crf_lr, 'weight_decay':1e-2, 'params': []}
        for name, par in self.model.named_parameters():
            if 'crf' in name or name == 'fc':
                params2['params'].append(par)
            else:
                params1['params'].append(par)
        parameters_to_optimize.append(params1)
        parameters_to_optimize.append(params2)

        self.optimizer = optim.AdamW(parameters_to_optimize)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
                                                            num_warmup_steps=self.args.warmup_ratio*self.train_num_steps, 
                                                                num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)


class NERTrainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None, processor=None, label_map=None, args=None, logger=None,  writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.processor = processor
        self.dev_dict = processor.load_from_file(mode="dev")
        self.test_dict = processor.load_from_file(mode="test")
        self.dev_sentences = self.test_dict['words']
        self.test_sentences = self.test_dict['words']
        self.dev_imgs = self.test_dict['imgs']
        self.test_imgs = self.test_dict['imgs']
        self.logger = logger
        self.label_map = label_map
        self.writer = writer
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_train_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.best_train_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs
        self.step = 0
        self.args = args

    def train(self):
        if self.args.use_prompt:
            self.multiModal_before_train()
        else:
            self.bert_before_train()

        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data)*self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        # if self.args.load_path is not None:  # load model from load_path
        #     self.logger.info("Loading model from {}".format(self.args.load_path))
        #     self.model.load_state_dict(torch.load(self.args.load_path))
        #     self.logger.info("Load model successful!")

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True, initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs+1):
                y_true, y_pred = [], []
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    attention_mask, labels, logits, loss = self._step(batch, mode="train")
                    avg_loss += loss.detach().cpu().item()

                    loss.backward()
                    if self.args.max_norm > 1:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_norm, norm_type=2)
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    

                    if isinstance(logits, torch.Tensor):    # CRF return lists
                        logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                    label_ids = labels.to('cpu').numpy()
                    input_mask = attention_mask.to('cpu').numpy()
                    label_map = {idx:label for label, idx in self.label_map.items()}
                    
                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_predict = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_map[label_ids[row][column]] != "X" and label_map[label_ids[row][column]] != "[SEP]":
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_predict.append(label_map[logits[row][column]])
                            else:
                                break
                        y_true.append(true_label)
                        y_pred.append(true_predict)

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss, global_step=self.step)    # tensorbordx
                        avg_loss = 0
                results = classification_report(y_true, y_pred, digits=4) 
                self.logger.info("***** Train Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[0].split('    ')[3])
                if self.writer:
                    self.writer.add_scalar(tag='train_f1', scalar_value=f1_score, global_step=epoch)    # tensorbordx
                self.logger.info("Epoch {}/{}, best train f1: {}, best epoch: {}, current train f1 score: {}."\
                            .format(epoch, self.args.num_epochs, self.best_train_metric, self.best_train_epoch, f1_score))
                if f1_score > self.best_train_metric:
                    self.best_train_metric = f1_score
                    self.best_train_epoch = epoch

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)   # generator to dev.

            torch.cuda.empty_cache()
            
            pbar.close()
            self.pbar = None
            self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch, self.best_dev_metric))
            self.logger.info("Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch, self.best_test_metric))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        y_true, y_pred = [], []
        step = 0
        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    attention_mask, labels, logits, loss = self._step(batch, mode="dev")    # logits: batch, seq, num_labels
                    total_loss += loss.detach().cpu().item()

                    if isinstance(logits, torch.Tensor):
                        logits = logits.argmax(-1).detach().cpu().numpy()  # batch, seq, 1
                    label_ids = labels.detach().cpu().numpy()
                    input_mask = attention_mask.detach().cpu().numpy()
                    label_map = {idx:label for label, idx in self.label_map.items()}
                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_predict = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_map[label_ids[row][column]] != "X" and label_map[label_ids[row][column]] != "[SEP]":
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_predict.append(label_map[logits[row][column]])
                            else:
                                break
                        y_true.append(true_label)
                        y_pred.append(true_predict)

                    pbar.update()
                pbar.close()
                results = classification_report(y_true, y_pred, digits=4)  
                self.logger.info("***** Dev Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1]) 
                if self.writer:
                    self.writer.add_scalar(tag='dev_f1', scalar_value=f1_score, global_step=epoch)    # tensorbordx
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss/step, global_step=epoch)    # tensorbordx

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}."\
                            .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch, f1_score))
                if f1_score >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = f1_score # update best metric(f1 score)
                    # if self.args.save_path is not None:
                    torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, "best_model.pth"))
                    self.logger.info("Save best model at {}".format(self.args.output_dir))
                    self._write_results(os.path.join(self.args.output_dir, "dev_best_pred.txt"), y_pred, y_true, self.dev_sentences, self.dev_imgs)
                    self.logger.info("Write best pred as {}".format(self.args.output_dir))


        self.model.train()

    def test(self):
        self.model.to(self.args.device)
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data)*self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

   
        self.logger.info("Loading model from {}".format(self.args.output_dir))
        self.model.load_state_dict(torch.load( os.path.join(self.args.output_dir, "best_model.pth")))
        self.logger.info("Load model successful!")
        y_true, y_pred = [], []
        entity_type = "PER"
        entity_reps, image_reps = [], []
        size = 0
        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device)  if isinstance(tup, torch.Tensor) else tup for tup in batch)  # to cpu/cuda device
                    attention_mask, labels, logits, loss, ent_reps, img_reps = self._step(batch, mode="dev")    # logits: batch, seq, num_labels
                    if len(ent_reps) > 0:
                        entity_reps.append(ent_reps)
                        image_reps.append(img_reps)
                        size = size + ent_reps.shape[0]
                        # pdb.set_trace()
                        if size >= 150:
                            pdb.set_trace()
                            entity_reps = torch.cat(entity_reps, dim=0)
                            image_reps = torch.cat(image_reps, dim=0)
                            entity_reps_npy = entity_reps.cpu().detach().numpy()
                            image_reps_npy = image_reps.cpu().detach().numpy()
                            txt_filename = "./reps/" + self.args.dataset_name + "_" + entity_type + "_txt.npy"
                            img_filename = "./reps/" + self.args.dataset_name + "_" + entity_type + "_img.npy"
                            np.save(txt_filename,entity_reps_npy)
                            np.save(img_filename, image_reps_npy)
                            break
                    total_loss += loss.detach().cpu().item()

                    if isinstance(logits, torch.Tensor): 
                        logits = logits.argmax(-1).detach().cpu().tolist()  # batch, seq, 1
                    label_ids = labels.detach().cpu().numpy()
                    input_mask = attention_mask.detach().cpu().numpy()
                    label_map = {idx:label for label, idx in self.label_map.items()}
                    for row, mask_line in enumerate(input_mask):
                        true_label = []
                        true_predict = []
                        for column, mask in enumerate(mask_line):
                            if column == 0:
                                continue
                            if mask:
                                if label_map[label_ids[row][column]] != "X" and label_map[label_ids[row][column]] != "[SEP]":
                                    true_label.append(label_map[label_ids[row][column]])
                                    true_predict.append(label_map[logits[row][column]])
                            else:
                                break
                        y_true.append(true_label)
                        y_pred.append(true_predict)
                    pbar.update()
                # evaluate done
                pbar.close()
                results = classification_report(y_true, y_pred, digits=4) 

                self.logger.info("***** Test Eval results *****")
                self.logger.info("\n%s", results)
                f1_score = float(results.split('\n')[-4].split('      ')[-2].split('    ')[-1])
                if self.writer:
                    self.writer.add_scalar(tag='test_f1', scalar_value=f1_score)    # tensorbordx
                    self.writer.add_scalar(tag='test_loss', scalar_value=total_loss/len(self.test_data))    # tensorbordx
                total_loss = 0
                self.logger.info("Test f1 score: {}.".format(f1_score))

                self._write_results(os.path.join(self.args.output_dir, "test_best_pred.txt"), y_pred, y_true, self.test_sentences, self.test_imgs)
                    
        self.model.train()

    def _write_results(self, output_pred_file, y_pred, y_true, sentences, imgs):
        # sentence_list = []
        fout = open(output_pred_file, 'w')
        for i in range(len(y_pred)):
            sentence = sentences[i]
            # sentence_list.append(sentence)
            img = imgs[i]
            samp_pred_label = y_pred[i]
            samp_true_label = y_true[i]
            fout.write(img+'\n')
            fout.write(' '.join(sentence)+'\n')
            fout.write(' '.join(samp_pred_label)+'\n')
            fout.write(' '.join(samp_true_label)+'\n'+'\n')
        fout.close()
        
    def _step(self, batch, mode="train"):
        # pdb.set_trace()
        if self.args.use_prompt:
            input_ids, token_type_ids, attention_mask, labels, images, aux_imgs = batch
        else:
            images, aux_imgs = None, None
            input_ids, token_type_ids, attention_mask, labels = batch
        output, dis_loss, txt_kdl, img_kdl, ent_reps, img_reps = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, images=images, aux_imgs=aux_imgs, mode=mode)
        logits, crf_loss = output.logits, output.loss
        if mode == 'train':
            if self.step % 100 == 0:
                self.logger.info("crf_loss: {}, dis_loss: {}, txt_loss: {}, img_kdl: {}".format(crf_loss.item(), dis_loss.item(), txt_kdl.item(), img_kdl.item()))
        loss = crf_loss + self.args.beta1 * dis_loss + self.args.beta2 * txt_kdl + self.args.beta3 * img_kdl
        # loss = dis_loss
        return attention_mask, labels, logits, loss, ent_reps, img_reps



    def bert_before_train(self):
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr)

        self.model.to(self.args.device)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
                                                            num_warmup_steps=self.args.warmup_ratio*self.train_num_steps, 
                                                            num_training_steps=self.train_num_steps)

    def multiModal_before_train(self):

        # self.vis_encoding = ImageModel() 
        # self.num_labels  = len(label_list)  # pad
        # self.crf = CRF(self.num_labels, batch_first=True)
        # self.fc = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        # self.dropout = nn.Dropout(0.1)

        # self.img2txt_attention = CrossAttention(heads=12, hidden_size = args.hidden_size)
        # self.txt2txt_attention = CrossAttention(heads=12, hidden_size = args.hidden_size)
        # self.vis2text = nn.Linear(2048, args.hidden_size)
        # self.linear = nn.Linear(args.hidden_size * 2, args.hidden_size)

        # if self.args.use_bias:
        #     self.img2txt_bias_layer = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
        #     self.txt2img_bias_layer = nn.Bilinear(self.hidden_size, self.hidden_size, 1)
        
        # bert_lr
        # parameters = []
        # params1 = {'lr':self.args.bert_lr, 'weight_decay':1e-2, 'params': []}
        # params2 = {'lr':self.args.lr, 'weight_decay':1e-2, 'params': []}
        # for name, param in self.model.named_parameters():
        #     if 'bert' in name:
        #         params1['params'].append(param)
        #     else:
        #         params2['params'].append(param)
        # parameters.append(params1)
        # parameters.append(params2)


        # bert lr
        # parameters = []
        # params = {'lr':self.args.lr, 'weight_decay':1e-2}
        # params['params'] = []
        # for name, param in self.model.named_parameters():
        #     if 'bert' in name:
        #         params['params'].append(param)
        # parameters.append(params)

        # # prompt lr
        # params = {'lr':self.args.lr, 'weight_decay':1e-2}
        # params['params'] = []
        # for name, param in self.model.named_parameters():
        #     if 'encoder_conv' in name or 'gates' in name:
        #         params['params'].append(param)
        # parameters.append(params)

        # # crf lr
        # params = {'lr':5e-2, 'weight_decay':1e-2}
        # params['params'] = []
        # for name, param in self.model.named_parameters():
        #     if 'crf' in name or name.startswith('fc'):
        #         params['params'].append(param)
        
        if self.args.version == 'old':
            if not self.args.tune_resnet:
                for name, par in self.model.named_parameters(): # freeze resnet
                    if 'vis_encoding' in name:   par.requires_grad = False

            parameters_to_optimize = list(self.model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            parameters_to_optimize = [ 
                {'params': [p for n, p in parameters_to_optimize 
                    if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, 'lr':self.args.lr},                                                                                                                         
                {'params': [p for n, p in parameters_to_optimize
                    if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, 'lr':self.args.lr}
            ]
        else:
            if not self.args.tune_resnet:
                for name, par in self.model.named_parameters(): # freeze resnet
                    if 'image_model' in name:   par.requires_grad = False

            parameters_to_optimize = []
            params1 = {'lr':self.args.lr, 'weight_decay':1e-2, 'params': []}
            params2 = {'lr':self.args.crf_lr, 'weight_decay':1e-2, 'params': []}
            for name, par in self.model.named_parameters():
                if 'crf' in name or name == 'fc':
                    params2['params'].append(par)
                else:
                    params1['params'].append(par)
            parameters_to_optimize.append(params1)
            parameters_to_optimize.append(params2)

            # parameters_to_optimize = []
            # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            # params0 = {'lr':self.args.lr, 'weight_decay':0.0, 'params': []}
            # params1 = {'lr':self.args.lr, 'weight_decay':1e-2, 'params': []}
            # params2 = {'lr':self.args.crf_lr, 'weight_decay':1e-2, 'params': []}
            # for name, par in self.model.named_parameters():
            #     if ('crf' in name) or (name == 'fc'):
            #         params2['params'].append(par)
            #     else:
            #         if not any(nd in name for nd in no_decay):
            #             params1['params'].append(par)
            #         if any(nd in name for nd in no_decay):
            #             params0['params'].append(par)
            # parameters_to_optimize.append(params1)
            # parameters_to_optimize.append(params2)
            # parameters_to_optimize.append(params0)


        self.optimizer = optim.AdamW(parameters_to_optimize)
        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer, 
                                                            num_warmup_steps=self.args.warmup_ratio*self.train_num_steps, 
                                                            num_training_steps=self.train_num_steps)
        self.model.to(self.args.device)
