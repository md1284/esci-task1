import os
import re
import sys
import json
import uuid
import copy
import torch
import string
import pickle
import numpy 
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.distributed as dist

import random

import time
import torch.nn.functional as F

from transformers import Adafactor
from transformers import (
    T5Config,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from torch.utils.data import DataLoader
from itertools import chain

from data import GENREDataset, GENRETriplesDataset, pickle_dump, pickle_load, psg_to_pids


class T5BaseClass(pl.LightningModule):
    def __init__(self):
        super(T5BaseClass, self).__init__()
        
        if torch.cuda.current_device() == 0:
            self.print = True 
        else:
            self.print = False 

    def _get_dataset(self, split):
        if self.hparams.model_mode == "gr":
            dataset = GENREDataset(
                tokenizer=self.tokenizer,
                split=split,
                hparams=self.hparams,
            )
        elif self.hparams.model_mode == "grcl":
            dataset = GENRETriplesDataset(
                tokenizer=self.tokenizer,
                split=split,
                hparams=self.hparams,
            )
        return dataset

    def train_dataloader(self):
        train_dataset = self._get_dataset(split="train")
        dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.hparams.train_batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def val_dataloader(self):
        val_dataset = self._get_dataset(split="validation")
        dataloader = DataLoader(
            val_dataset,
            shuffle=False,
            batch_size=self.hparams.eval_batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def test_dataloader(self):
        test_dataset = self._get_dataset(split="test")
        dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=self.hparams.test_batch_size,
            drop_last=False,
            num_workers=self.hparams.num_workers,
        )
        return dataloader

    def _gather_object(self, obj):
        gathered = [None for _ in range(self.hparams.n_gpu)]
        dist.all_gather_object(gathered, obj)
        return gathered

    def gather_list(self, obj):
        gathered = self._gather_object(obj)
        output = []
        output = list(chain(*gathered))
        return output

    def normalize_answer(self, s):
        def remove_articles(text):
            return re.sub(r"\b(a|an|the)\b", " ", text)

        def white_space_fix(text):
            return " ".join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return "".join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def _calculate_recall(self, pred, gt):
        assert len(pred) == self.hparams.ret_num, f'gt: {gt}\npred: {pred}' 
        if type(gt) == list:
            gt = [self.normalize_answer(el) for el in gt]
            for elem in pred:
                if elem is None: continue 
                if self.normalize_answer(elem) in gt:
                    return 100
        else:
            for elem in pred:
                if elem is None: continue 
                if self.normalize_answer(elem) == self.normalize_answer(gt):
                    return 100
        return 0

    def _calculate_em(self, pred, gt):
        if pred is None: 
            return 0
        if type(gt) == list:
            gt = [self.normalize_answer(el) for el in gt]
            if self.normalize_answer(pred) in gt:
                return 100
            else:
                return 0
        else:
            if self.normalize_answer(pred) == self.normalize_answer(gt):
                return 100
            else:
                return 0

    def lmap(self, f, x):
        return list(map(f, x))

class T5FineTuner(T5BaseClass):
    def __init__(self, args):
        super(T5FineTuner, self).__init__()
        
        self.save_hyperparameters(args)

        if self.hparams.do_train:
            self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
            self.tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.model_name_or_path}")
        elif self.hparams.do_test:
            self.model = T5ForConditionalGeneration.from_pretrained(args.test_model_path)
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
            if self.print:
                print(f"@@@ Loading Test Model from {self.hparams.test_model_path}")

        self.qid_to_trie = pickle_load(args.qid_to_trie)
        self.qid_to_queryid = pickle_load(os.path.join(args.dataset, args.qid_to_queryid))
        self.qid_to_candidate_pid = pickle_load(os.path.join(args.dataset, args.qid_to_candidate_pid))
        self.qid_to_esci_pid = pickle_load(os.path.join(args.dataset, args.qid_to_esci_pid))
        self.pid_to_productid = pickle_load(os.path.join(args.dataset, args.pid_to_productid))
        self.pid_to_product = pickle_load(os.path.join(args.dataset, args.pid_to_product))
        self.psg_to_pids, self.pid_to_psg = psg_to_pids(os.path.join(args.dataset, args.collection))

        self.ndcg_score_list = []
        self.test_queryid_list = []
        self.test_productids_list = []
        self.qid_pid_loss = []
        self.save_epoch = []

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_attention_mask=None,
        decoder_input_ids=None,
        lm_labels=None,
        return_dict=True
    ):
        if lm_labels is None:
            assert decoder_input_ids is not None
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=return_dict,
            )
        if decoder_input_ids is None:
            assert lm_labels is not None
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=lm_labels,
                return_dict=return_dict
            )
    
    def _loss(self, batch):
        lm_labels = copy.deepcopy(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )
        loss = outputs[0]
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log(
            "train loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss

    def _get_from_trie(self, input_ids, trie_dict, qid):
        #if qid == 57:
        #    print(f"input_ids: {input_ids}, trie_dict: {trie_dict}")
        if len(input_ids) == 0:
            possible_ids = list(trie_dict.keys())
            return possible_ids
        else:
            #assert input_ids[0] in trie_dict.keys(), input_ids # if score == -np.inf, then it will be neglected 
            if input_ids[0] in trie_dict.keys():
                return self._get_from_trie(input_ids[1:], trie_dict[input_ids[0]], qid)
            else:
                return []
    
    def get(self, qid, batch_id, input_ids):
        assert input_ids[0] == 0
        trie = self.qid_to_trie[qid]
        next_ids = self._get_from_trie(input_ids, trie, qid)
        return next_ids

    def ids_to_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return self.lmap(str.strip, gen_text)

    def _get_ids(self, s, max_length):
        ids = self.tokenizer.batch_encode_plus(
            [s],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return ids

    def dcg(self, pred_items, gt_items, k):
        assert len(pred_items) == k
        s_dcg = []
        for i, pred_item in enumerate(pred_items):
            if pred_item in gt_items:
                s_dcg.append(np.reciprocal(np.log2(i + 2)))
        return sum(s_dcg)

    def idcg(self, gt_items, k):
        s_idcg = [np.reciprocal(np.log2(i + 2)) for i in range(min(len(gt_items), k))]
        return sum(s_idcg)

    def ndcg(self, pred_items, gt_items, k=10):
        s_dcg = self.dcg(pred_items[:k], gt_items, k=k)
        s_idcg = self.idcg(gt_items, k=k)
        return 100 * s_dcg / s_idcg

    def calculate_scores(self, generated_pids, esci_dicts):
        ndcg_list = []
        for gen_pids, esci_dict in zip(generated_pids, esci_dicts):
            E_pids = esci_dict["E"]
            ndcg = self.ndcg(gen_pids, E_pids, k=self.hparams.val_beam_size)
            ndcg_list.append(ndcg)
        return ndcg_list

    def _val_step(self, batch, batch_idx, isTest=False, return_elem=False):
        # calculates recall and em -> returns the list of each score
        query_ids = [self.qid_to_queryid[qid.item()] for qid in batch["qid"]]
        
        _generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            use_cache=True,
            max_length=self.hparams.max_beamsearch_length,
            num_beams=self.hparams.val_beam_size,
            num_return_sequences=self.hparams.val_beam_size,
            prefix_allowed_tokens_fn=lambda batch_id, sent: self.get(
                batch["qid"][batch_id].item(), batch_id, sent.tolist()
            ),
            early_stopping=True,
        )
        _generated_text = self.ids_to_text(_generated_ids)
        inum = len(_generated_ids) // self.hparams.val_beam_size
        assert inum == len(batch["output"])
        generated_text = [
            _generated_text[
                i * self.hparams.val_beam_size : (i + 1) * self.hparams.val_beam_size
            ]
            for i in range(inum)
        ]
        generated_ids = [
            _generated_ids[
               i * self.hparams.val_beam_size : (i+1) * self.hparams.val_beam_size
            ].detach().cpu().numpy().tolist()
            for i in range(inum)
        ]

        """
        print(f"generated_text: {generated_text}")
        generated_ids = []
        reference_ids = []
        for batch_id, qid in enumerate(batch["qid"]):
            qid = qid.item()
            reference_pids = self.qid_to_esci_pid[qid]["E"]
            reference_texts = [self.pid_to_product[pid] for pid in reference_pids]
            batch_reference_ids = [self.tokenizer(text)["input_ids"] for text in reference_texts]
            reference_ids.append(batch_reference_ids)

            batch_generated_ids = [self.tokenizer(text)["input_ids"] for text in generated_text[batch_id]]
            generated_ids.append(generated_ids)

        print(f"generated_ids: {generated_ids[:5]}")
        print(f"reference_ids: {reference_ids}")
        assert len(generated_ids) == len(reference_ids)

        ndcg_list = []
        for gen_ids, ref_ids in zip(generated_ids, reference_ids):
            ndcg = self.ndcg(gen_ids, ref_ids, k=self.hparams.val_beam_size)
            ndcg_list.append(ndcg)
        """

        generated_pids = []
        candidate_counts = []
        for batch_id, texts in enumerate(generated_text):
            candiate_pids = set(self.qid_to_candidate_pid[batch["qid"][batch_id].item()])
            candidate_counts.append(len(candiate_pids))
            generated_pid = []
            for text in texts:
                #print(f"text: {text}")
                pids = set(self.psg_to_pids[text])
                #print(f"pids: {pids}")
                inter = list(candiate_pids & pids)

                sample = self.pid_to_psg[inter[0]]
                for pid in inter:
                    assert sample == self.pid_to_psg[pid], inter
                generated_pid.extend(inter)
            generated_pids.append(generated_pid)

        esci_dicts = [self.qid_to_esci_pid[qid.item()] for qid in batch["qid"]]
        ndcg_list = self.calculate_scores(generated_pids, esci_dicts)

        if isTest:
            generated_productids_list = []
            for pids, candidate_count in zip(generated_pids, candidate_counts):
                generated_productids = []
                for pid in pids:
                    if len(generated_productids) == candidate_count:
                        break
                    if self.pid_to_productid[pid] in set(generated_productids):
                        continue
                    generated_productids.append(self.pid_to_productid[pid])
                generated_productids_list.append(generated_productids)
            return {
                "query_ids": query_ids,
                "generated_pids": list(generated_pids),
                "generated_productids_list": list(generated_productids_list),
            }

        if return_elem:
            assert (
                len(list(batch["input"]))
                == len(list(generated_pids))
                == len(list(ndcg_list))
            )
            return {
                "input": list(batch["input"]),
                "pred": list(generated_text),
                "gt": list(esci_pid),
                "ndcg_list": list(ndcg_list),
            }
        else:
            return ndcg_list
    
    def validation_step(self, batch, batch_idx):
        ndcg_score = self._val_step(batch, batch_idx)
        self.ndcg_score_list.extend(list(ndcg_score))

    def validation_epoch_end(self, outputs):
        avg_ndcg = np.mean(np.array(self.ndcg_score_list))
        self.ndcg_score_list = []
        self.log(
            "val ndcg",
            avg_ndcg,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return

    def _prediction(self, batch, batch_idx):
        assert len(batch["qid"]) == 1, len(batch["qid"])
        qid = batch["qid"].item()
        pid = batch["pid"].item()
        loss = self._loss(batch).item()
        qid_pid_loss = [[qid, pid, loss]]
        return qid_pid_loss


    def test_step(self, batch, batch_idx):
        #ret_dict = self._val_step(batch, batch_idx, isTest=True, return_elem=True)
        #self.test_queryid_list.extend(ret_dict["query_ids"])
        #self.test_productids_list.extend(ret_dict["generated_productids_list"])
        #self._save_test()
        qid_pid_loss = self._prediction(batch, batch_idx)
        self.qid_pid_loss.extend(qid_pid_loss)

        if batch_idx % 10000 == 0:
            self._save_loss()
    
    def test_epoch_end(self, outputs):
        #self._save_test(epoch_end=True)
        self._save_loss(epoch_end=True)

    def _save_loss(self, epoch_end=False):
        os.makedirs(self.hparams.output_dir, exist_ok=True)
        qid_pid_loss = self.gather_list(self.qid_pid_loss)
        
        if self.print:
            with open(os.path.join(self.hparams.output_dir, self.hparams.test_name), "w") as f:
                for qid, pid, loss in self.qid_pid_loss:
                    query_id = self.qid_to_queryid[qid]
                    product_id = self.pid_to_productid[pid]
                    f.write(f"{query_id} Q0 {product_id} {loss} {self.hparams.test_name}\n")

            if epoch_end:
                print(
                    f"Saving in {self.hparams.test_name}!\nnumber of elements: {len(qid_pid_loss)}"
                )

    def _save_test(self, epoch_end=False):
        os.makedirs(self.hparams.output_dir, exist_ok=True)

        query_ids = self.gather_list(self.test_queryid_list)
        product_ids_list = self.gather_list(self.test_productids_list)
        
        if self.print:
            with open(os.path.join(self.hparams.output_dir, self.hparams.test_name), "w") as f:
                for query_id, product_ids in zip(query_ids, product_ids_list):
                    for rank, product_id in enumerate(product_ids):
                        f.write(f"{query_id} Q0 {product_id} {rank} {round(1/(rank+1),3)} {self.hparams.test_name}\n")
            if epoch_end:
                print(
                    f"Saving in {self.hparams.test_name}!\nnumber of elements: {len(query_ids)}"
                )

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            warmup_init=False,
            scale_parameter=False,
            relative_step=False,
        )
        self.opt = optimizer
        if self.hparams.lr_scheduler == "constant":
            return [optimizer]
        elif self.hparams.lr_scheduler == "exponential":
            len_data = len(self.train_dataloader())
            denominator = self.hparams.n_gpu
            steps_per_epoch = (
                (len_data // denominator) + 1
            ) // self.hparams.gradient_accumulation_steps
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                epochs=self.hparams.num_train_epochs,
                anneal_strategy="linear",
                cycle_momentum=False,
            )
            return [optimizer], [
                {"scheduler": scheduler, "interval": "step", "name": "learning_rate"}
            ]
        else:
            raise NotImplementedError("Choose lr_schduler from (constant|exponential)")

    def on_save_checkpoint(self, checkpoint):
        save_path = os.path.join(
            self.hparams.output_dir, f"best_tfmr_{self.current_epoch}"
        )
        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.save_epoch.append(self.current_epoch)
        except:
            print(f"Fail to save ... {self.current_epoch}")
        if len(self.save_epoch) > 5:
            rm_file = f"best_tfmr_{self.save_epoch[0]}"
            os.system(f"rm -rf {os.path.join(self.hparams.output_dir, rm_file)}")


class T5NegFineTuner(T5BaseClass):
    def __init__(self, args):
        super(T5NegFineTuner, self).__init__()
        self.save_hyperparameters(args)

        self.config = T5Config.from_pretrained(args.model_name_or_path)
        self.config.update({"output_hidden_states": True})
        if self.hparams.do_train:
            self.model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path, config=self.config)
            self.tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
            if self.print:
                print(f"@@@ Loading Model from {self.hparams.model_name_or_path}")
        elif self.hparams.do_test:
            self.model = T5ForConditionalGeneration.from_pretrained(args.test_model_path, config=self.config)
            self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.test_model_path)
            if self.print:
                print(f"@@@ Loading Test Model from {self.hparams.test_model_path}")

        self.qid_to_trie = pickle_load(args.qid_to_trie)
        self.qid_to_queryid = pickle_load(os.path.join(args.dataset, args.qid_to_queryid))
        self.qid_to_candidate_pid = pickle_load(os.path.join(args.dataset, args.qid_to_candidate_pid))
        self.qid_to_esci_pid = pickle_load(os.path.join(args.dataset, args.qid_to_esci_pid))
        self.pid_to_productid = pickle_load(os.path.join(args.dataset, args.pid_to_productid))
        self.pid_to_product = pickle_load(os.path.join(args.dataset, args.pid_to_product))
        self.psg_to_pids, self.pid_to_psg = psg_to_pids(os.path.join(args.dataset, args.collection))

        self.ndcg_score_list = []
        self.test_queryid_list = []
        self.test_productids_list = []
        self.qid_pid_loss = []
        self.save_epoch = []

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduce=False)

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_attention_mask=None,
        decoder_input_ids=None,
        lm_labels=None,
        return_dict=True
    ):
        if lm_labels is None:
            assert decoder_input_ids is not None
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                return_dict=return_dict,
            )
        if decoder_input_ids is None:
            assert lm_labels is not None
            return self.model(
                input_ids,
                attention_mask=attention_mask,
                decoder_attention_mask=decoder_attention_mask,
                labels=lm_labels,
                return_dict=return_dict
            )
        
    def flatten(self, L):
        # return [x for y in L for x in y]
        result = []
        for _list in L:
            result += _list

        return result

    def _loss(self, batch):
        lm_labels = copy.deepcopy(batch["target_ids"])
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )
        loss = outputs[0]
        return loss

    def _colbert_score(self, Q, D):
        scores = D @ Q.to(dtype=D.dtype).permute(0,2,1)
        scores = scores.max(1).values # (bsize, Q_len)
        scores = scores.sum(-1) # (bsize)
        return scores
    
    def compute_ib_loss(self, Q, D):
        scores = (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)  # query-major unsqueeze
        scores = scores.max(1).values
        scores = scores.sum(-1)

        nway = 2
        all_except_self_negatives = [list(range(qidx*D.size(0), qidx*D.size(0) + nway*qidx+1)) +
                                     list(range(qidx*D.size(0) + nway * (qidx+1), qidx*D.size(0) + D.size(0)))
                                     for qidx in range(Q.size(0))]

        scores = scores[self.flatten(all_except_self_negatives)]
        scores = scores.view(Q.size(0), -1)

        labels = torch.arange(0, Q.size(0), device=scores.device) * nway
        return nn.CrossEntropyLoss()(scores, labels)

    def colbert_loss(self, pos_outputs, neg_outputs, source_mask, pos_target_mask, neg_target_mask, return_scores=False):
        Q = pos_outputs["outputs"].inputs_embeds
        #Q = pos_outputs["outputs"].encoder_last_hidden_state
        P = pos_outputs["outputs"].sequence_output
        N = neg_outputs["outputs"].sequence_output

        #Q = F.normalize(Q, dim=-1)
        #P = F.normalize(P, dim=-1)
        #N = F.normalize(N, dim=-1)

        Q = Q * source_mask.unsqueeze(dim=-1)
        P = P * pos_target_mask.unsqueeze(dim=-1)
        N = N * neg_target_mask.unsqueeze(dim=-1)

        pos_scores = self._colbert_score(Q, P)
        neg_scores = self._colbert_score(Q, N)
        pos_scores = pos_scores.unsqueeze(dim=-1)
        neg_scores = neg_scores.unsqueeze(dim=-1)

        scores = torch.cat([pos_scores, neg_scores], dim=-1)
        if return_scores:
            return scores

        labels = torch.zeros(self.hparams.train_batch_size, dtype=torch.long, device=scores.device)
        loss = nn.CrossEntropyLoss(ignore_index=-100)(scores, labels[:scores.size(0)])

        if self.hparams.use_ib_negatives:
            D = [torch.stack([p,n]) for p, n in zip(P, N)]
            D = torch.cat(D)
            ib_loss = self.compute_ib_loss(Q, D)
            return loss + ib_loss

        return loss

    def pos_loss(self, batch, swap=False, return_elem=False):
        source_ids, source_mask, pos_target_ids, pos_target_mask = \
            batch["source_ids"], batch["source_mask"], batch["pos_target_ids"], batch["pos_target_mask"]
        if swap:
            pos_target_ids, pos_target_mask, source_ids, source_mask = \
                    source_ids, source_mask, pos_target_ids, pos_target_mask

        lm_labels = copy.deepcopy(pos_target_ids)
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=source_ids,
            attention_mask=source_mask,
            lm_labels=lm_labels,
            decoder_attention_mask=pos_target_mask,
        )
        if return_elem:
            return {
                "outputs": outputs,
                "lm_labels": lm_labels
            }

        loss = outputs[0]
        return loss

    def neg_loss(self, batch, swap=False, return_elem=False):
        source_ids, source_mask, neg_target_ids, neg_target_mask = \
            batch["source_ids"], batch["source_mask"], batch["neg_target_ids"], batch["neg_target_mask"]
        if swap:
            neg_target_ids, neg_target_mask, source_ids, source_mask = \
                    source_ids, source_mask, neg_target_ids, neg_target_mask

        lm_labels = copy.deepcopy(neg_target_ids)
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=source_ids,
            attention_mask=source_mask,
            lm_labels=lm_labels,
            decoder_attention_mask=neg_target_mask,
        )
        if return_elem:
            return {
                "outputs": outputs,
                "lm_labels": lm_labels
            }

        loss = outputs[0]
        return loss

    def lul(self, pos_outputs, neg_outputs):
        pos_logits, pos_lables = pos_outputs["outputs"].logits, pos_outputs["lm_labels"]
        neg_logits, neg_labels = neg_outputs["outputs"].logits, neg_outputs["lm_labels"]

        pos_loss = self.loss_fct(pos_logits.view(-1, pos_logits.size(-1)), pos_lables.view(-1)).view(pos_lables.size(0), -1)
        neg_loss = self.loss_fct(neg_logits.view(-1, neg_logits.size(-1)), neg_labels.view(-1)).view(neg_labels.size(0), -1)
        pos_logprob = torch.mean(-pos_loss, dim=-1)
        neg_logprob = torch.mean(-neg_loss, dim=-1)

        outputs = pos_logprob + torch.log(1 - torch.exp(neg_logprob) + 1e-20)
        loss = - torch.mean(outputs)
        return loss


    def training_step(self, batch, batch_idx):
        #loss = self._loss(batch)
        #swap = self.hparams.swap
        swap = False
        if random.randint(0,1) == 1:
            swap = True

        pos_outputs = self.pos_loss(batch, swap=swap, return_elem=True)
        neg_outputs = self.neg_loss(batch, swap=swap, return_elem=True)

        source_mask, pos_target_mask, neg_target_mask = batch["source_mask"], batch["pos_target_mask"], batch["neg_target_mask"]
        if swap:
            pos_target_mask, neg_target_mask = neg_target_mask, pos_target_mask

        if swap:
            # LUL
            loss = self.lul(pos_outputs, neg_outputs)
        else:
            # colbert
            loss = self.colbert_loss(pos_outputs, neg_outputs, source_mask, pos_target_mask, neg_target_mask)
        
        """
        loss_lul = self.lul(pos_outputs, neg_outputs)
        loss_colbert = self.colbert_loss(pos_outputs, neg_outputs, source_mask, pos_target_mask, neg_target_mask)
        loss = loss_lul + loss_colbert
        """

        # pos - neg
        """
        pos_loss = self.pos_loss(batch)
        neg_loss = self.neg_loss(batch)
        loss = pos_loss - neg_loss
        """

        self.log(
            "train loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        return loss

    def get_weight(self, label):
        if label == 4:
            return 1
        if label == 3:
            return 0.1
        if label == 2:
            return 0.01
        if label == 1:
            return 0
        raise NotImplementedError

    def dcg(self, pred_pids, gt_dict, k=None):
        dcg = []
        if k is None:
            k = len(pred_pids)

        for i, candidate in enumerate(pred_pids):
            if i == k:
                break
            assert candidate in gt_dict, candidate
            label = gt_dict[candidate]
            weight = self.get_weight(label)
            dcg.append(weight * np.reciprocal(np.log2(i + 2)))
        
        return sum(dcg)

    def idcg(self, gt_dict, k=None):
        idcg_score = []
        sorted_gt_dict = sorted(gt_dict.items(), key=lambda x: x[1], reverse=True)

        if k is None:
            k = len(sorted_gt_dict)

        for i, (product_id, label) in enumerate(sorted_gt_dict):
            if i == k:
                break
            weight = self.get_weight(label)
            idcg_score.append(weight * np.reciprocal(np.log2(i + 2)))

        return sum(idcg_score)

    def calculate_scores(self, generated_pids, esci_dicts):
        ndcg_list = []
        for gen_pids, esci_dict in zip(generated_pids, esci_dicts):
            reference_dict = {}
            for E_pid in esci_dict["E"]:
                reference_dict[E_pid] = 4
            for S_pid in esci_dict["S"]:
                reference_dict[S_pid] = 3
            for C_pid in esci_dict["C"]:
                reference_dict[C_pid] = 2
            for I_pid in esci_dict["I"]:
                reference_dict[I_pid] = 1

            k = len(gen_pids)
            idcg_score = self.idcg(reference_dict, k=k)
            dcg_score = self.dcg(gen_pids, reference_dict, k=k)
            ndcg_list.append(dcg_score / idcg_score)

        return ndcg_list


    def _prediction(self, batch, batch_idx, swap=False):
        source_ids, source_mask, target_ids, target_mask = \
            batch["source_ids"], batch["source_mask"], batch["target_ids"], batch["target_mask"]
        if swap:
            target_ids, target_mask, source_ids, source_mask = source_ids, source_mask, target_ids, target_mask

        lm_labels = copy.deepcopy(target_ids)
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=source_ids,
            attention_mask=source_mask,
            lm_labels=lm_labels,
            decoder_attention_mask=target_mask,
        )
        lm_logits = outputs.logits
        
        if swap:
            # LUL
            loss = self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(lm_labels.size(0), -1)
            loss = torch.mean(loss, dim=-1)
        else:
            # colbert
            Q = outputs.inputs_embeds
            #Q = outputs.encoder_last_hidden_state
            D = outputs.sequence_output

            #Q = F.normalize(Q, dim=-1)
            #D = F.normalize(D, dim=-1)

            Q = Q * source_mask.unsqueeze(dim=-1)
            D = D * target_mask.unsqueeze(dim=-1)
            cb_scores = self._colbert_score(Q, D)
            loss = - cb_scores


        """
        loss_lul = self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)).view(lm_labels.size(0), -1)
        loss_lul = torch.mean(loss_lul, dim=-1)

        Q = outputs.inputs_embeds
        D = outputs.sequence_output
        Q = Q * source_mask.unsqueeze(dim=-1)
        D = D * target_mask.unsqueeze(dim=-1)
        cb_scores = self._colbert_score(Q, D)
        loss_colbert = - cb_scores

        loss = loss_lul + loss_colbert
        """
        
        # colbert
        """
        Q = outputs.inputs_embeds
        D = outputs.sequence_output

        #Q = F.normalize(Q, dim=-1)
        #D = F.normalize(D, dim=-1)

        Q = Q * source_mask.unsqueeze(dim=-1)
        D = D * target_mask.unsqueeze(dim=-1)
        cb_scores = self._colbert_score(Q, D)
        loss = - cb_scores
        """
        
        return loss.tolist()
    
    def validation_step(self, batch, batch_idx):
        #losses = self._prediction(batch, batch_idx, swap=self.hparams.swap)
        losses_q2d = self._prediction(batch, batch_idx, swap=False)
        losses_d2q = self._prediction(batch, batch_idx, swap=True)
        losses = losses_q2d + losses_d2q

        qid_pid_loss = []
        qids = batch["qid"].tolist() * 2
        pids = batch["pid"].tolist() * 2
        #qids = batch["qid"].tolist()
        #pids = batch["pid"].tolist()

        assert len(losses) == len(qids) == len(pids)
        for qid, pid, loss in zip(qids, pids, losses):
            qid_pid_loss.append([qid, pid, loss])
        
        self.qid_pid_loss.extend(qid_pid_loss)

    def validation_epoch_end(self, outputs):
        qid_pid_loss = self.gather_list(self.qid_pid_loss)
        print(f"len(qid_pid_loss): {len(qid_pid_loss)}")

        qid_to_list = {}
        qid_to_sorted_list = {}
        for qid, pid, loss in qid_pid_loss:
            if qid not in qid_to_list.keys():
                qid_to_list[qid] = []
            qid_to_list[qid].append([pid, loss])

        for qid, pid_loss in qid_to_list.items():
            assert qid not in qid_to_sorted_list.keys(), qid
            
            uniq_sorted_pid = {}
            for pid, loss in pid_loss:
                if pid not in uniq_sorted_pid.keys():
                    uniq_sorted_pid[pid] = []
                uniq_sorted_pid[pid].append(loss)

            sorted_pid_loss = []
            for pid, pid_loss in uniq_sorted_pid.items():
                #assert len(pid_loss) == 2, f"len: {len(pid_loss)}, qid: {qid}, pid: {pid}, pid_loss: {pid_loss}"
                loss = np.mean(pid_loss)
                sorted_pid_loss.append([pid, loss])

            sorted_pid_loss = sorted(sorted_pid_loss, key=lambda x: x[1])
            qid_to_sorted_list[qid] = sorted_pid_loss

        ndcg_list = []
        for qid, sorted_list in qid_to_sorted_list.items():
            esci_pid = self.qid_to_esci_pid[qid]
            sorted_list = [pid for pid, loss in sorted_list]
            ndcg = self.calculate_scores([sorted_list], [esci_pid])
            ndcg_list.extend(ndcg)
        
        avg_ndcg = sum(ndcg_list)/len(ndcg_list)
        print(f"ndcg: {avg_ndcg}")
        self.qid_pid_loss = []

        self.log(
            "val ndcg",
            avg_ndcg,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return

    def test_step(self, batch, batch_idx):
        #ret_dict = self._val_step(batch, batch_idx, isTest=True, return_elem=True)
        #self.test_queryid_list.extend(ret_dict["query_ids"])
        #self.test_productids_list.extend(ret_dict["generated_productids_list"])
        #self._save_test()
        #losses = self._prediction(batch, batch_idx)
        losses_q2d = self._prediction(batch, batch_idx, swap=False)
        losses_d2q = self._prediction(batch, batch_idx, swap=True)
        losses = losses_q2d + losses_d2q

        qid_pid_loss = []
        qids = batch["qid"].tolist() * 2
        pids = batch["pid"].tolist() * 2
        #qids = batch["qid"].tolist()
        #pids = batch["pid"].tolist()

        assert len(losses) == len(qids) == len(pids)
        for qid, pid, loss in zip(qids, pids, losses):
            qid_pid_loss.append([qid, pid, loss])
        
        self.qid_pid_loss.extend(qid_pid_loss)

        #if batch_idx % 10000 == 0:
        #    self._save_loss()
    
    def test_epoch_end(self, outputs):
        qid_pid_loss = self.gather_list(self.qid_pid_loss)
        print(f"len(qid_pid_loss): {len(qid_pid_loss)}")

        qid_to_list = {}
        qid_to_sorted_list = {}
        for qid, pid, loss in qid_pid_loss:
            if qid not in qid_to_list.keys():
                qid_to_list[qid] = []
            qid_to_list[qid].append([pid, loss])

        for qid, pid_loss in qid_to_list.items():
            assert qid not in qid_to_sorted_list.keys(), qid
            
            uniq_sorted_pid = {}
            for pid, loss in pid_loss:
                if pid not in uniq_sorted_pid.keys():
                    uniq_sorted_pid[pid] = []
                uniq_sorted_pid[pid].append(loss)

            sorted_pid_loss = []
            for pid, pid_loss in uniq_sorted_pid.items():
                #assert len(pid_loss) == 2, f"len: {len(pid_loss)}, qid: {qid}, pid: {pid}, pid_loss: {pid_loss}"
                loss = np.mean(pid_loss)
                sorted_pid_loss.append([pid, loss])

            sorted_pid_loss = sorted(sorted_pid_loss, key=lambda x: x[1])
            qid_to_sorted_list[qid] = sorted_pid_loss

        ndcg_list = []
        for qid, sorted_list in qid_to_sorted_list.items():
            esci_pid = self.qid_to_esci_pid[qid]
            sorted_list = [pid for pid, loss in sorted_list]
            ndcg = self.calculate_scores([sorted_list], [esci_pid])
            ndcg_list.extend(ndcg)
        
        avg_ndcg = sum(ndcg_list)/len(ndcg_list)
        print(f"ndcg: {avg_ndcg}")
        
        #self._save_test(epoch_end=True)
        self._save_loss(qid_to_sorted_list, epoch_end=True)

    def _save_loss(self, qid_to_sorted_list, epoch_end=False):
        os.makedirs(self.hparams.output_dir, exist_ok=True)
        #qid_pid_loss = self.gather_list(self.qid_pid_loss)
        
        if self.print:
            with open(os.path.join(self.hparams.output_dir, self.hparams.test_name), "w") as f:
                for qid, sorted_list in qid_to_sorted_list.items():
                    query_id = self.qid_to_queryid[qid]
                    for pid, loss in sorted_list:
                        product_id = self.pid_to_productid[pid]
                        f.write(f"{query_id} Q0 {product_id} {loss} {self.hparams.test_name}\n")

            if epoch_end:
                print(
                    f"Saving in {self.hparams.test_name}!\nnumber of elements: {len(qid_to_sorted_list)}"
                )

    def _save_test(self, epoch_end=False):
        os.makedirs(self.hparams.output_dir, exist_ok=True)

        query_ids = self.gather_list(self.test_queryid_list)
        product_ids_list = self.gather_list(self.test_productids_list)
        
        if self.print:
            with open(os.path.join(self.hparams.output_dir, self.hparams.test_name), "w") as f:
                for query_id, product_ids in zip(query_ids, product_ids_list):
                    for rank, product_id in enumerate(product_ids):
                        f.write(f"{query_id} Q0 {product_id} {rank} {round(1/(rank+1),3)} {self.hparams.test_name}\n")
            if epoch_end:
                print(
                    f"Saving in {self.hparams.test_name}!\nnumber of elements: {len(query_ids)}"
                )

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            warmup_init=False,
            scale_parameter=False,
            relative_step=False,
        )
        self.opt = optimizer
        if self.hparams.lr_scheduler == "constant":
            return [optimizer]
        elif self.hparams.lr_scheduler == "exponential":
            len_data = len(self.train_dataloader())
            denominator = self.hparams.n_gpu
            steps_per_epoch = (
                (len_data // denominator) + 1
            ) // self.hparams.gradient_accumulation_steps
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                epochs=self.hparams.num_train_epochs,
                anneal_strategy="linear",
                cycle_momentum=False,
            )
            return [optimizer], [
                {"scheduler": scheduler, "interval": "step", "name": "learning_rate"}
            ]
        else:
            raise NotImplementedError("Choose lr_schduler from (constant|exponential)")

    def on_save_checkpoint(self, checkpoint):
        save_path = os.path.join(
            self.hparams.output_dir, f"best_tfmr_{self.current_epoch}"
        )
        try:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            self.save_epoch.append(self.current_epoch)
        except:
            print(f"Fail to save ... {self.current_epoch}")
        if len(self.save_epoch) > 5:
            rm_file = f"best_tfmr_{self.save_epoch[0]}"
            os.system(f"rm -rf {os.path.join(self.hparams.output_dir, rm_file)}")
