import os
import sys
import torch
import pickle
import copy
import ujson
import pandas as pd
import random
random.seed(12345)

from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import Dataset

def pickle_dump(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
        
def pickle_load(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def lmap(f, x):
    return list(map(f, x))

def _get_ids(s, max_length, tokenizer):
    ids = tokenizer.batch_encode_plus(
        [s],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return ids

def ids_to_text(generated_ids, tokenizer):
        gen_text = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return lmap(str.strip, gen_text)

def psg_to_pids(path):
    psg_to_pids = {}
    pid_to_psg = {}
    white_psg = []

    with open(path) as f:
        #for line_idx, line in tqdm(enumerate(f)):
        for line_idx, line in enumerate(f):
            try:
                pid, passage, *_ = line.strip().split('\t')
            except:
                if len(line.strip().split('\t')) == 1:
                    pid = line.strip()
                    passage = ""
                    white_psg.append(pid)
                else:
                    assert False
            assert pid == 'id' or int(pid) == line_idx
            if passage not in psg_to_pids.keys():
                psg_to_pids[passage] = []
            psg_to_pids[passage].append(int(pid))
            pid_to_psg[int(pid)] = passage
    print(f"white_psg: {len(white_psg)}")
    return psg_to_pids, pid_to_psg

class GENREDataset(Dataset):
    def __init__(self, tokenizer, split, hparams):
        self.hparams = hparams
        self.split = split
        if split == "train":
            data_path = self.hparams.train_file
        elif split == "validation":
            data_path = self.hparams.dev_file
        elif split == "test":
            data_path = self.hparams.test_file
        else:
            raise NotImplementedError(f"Inappropriate split type: {split}")

        self.queries = self._load_queries(os.path.join(self.hparams.dataset, self.hparams.queries))
        self.collection = self._load_collection(os.path.join(self.hparams.dataset, self.hparams.collection))

        dataset = self._load_datapair(os.path.join(self.hparams.dataset, data_path))
        if split in ["validation", "test"]:
            self.dataset = {"input": [], "output": [], "type": []}
            for input_, output_, type_ in zip(dataset["input"], dataset["output"], dataset["type"]):
                
                if split == "validation":
                    if input_ in self.dataset["input"]:
                        continue
                self.dataset["input"].append(input_)
                self.dataset["output"].append(output_)
                self.dataset["type"].append(type_)
        else:
            self.dataset = dataset
        self.dataset = pd.DataFrame(self.dataset)
        self.len = len(self.dataset)

        if torch.cuda.current_device() == 0:
            print(
                f"@@@ Loading from {os.path.join(self.hparams.dataset, data_path)}: {self.len}"
            )

        self.tokenizer = tokenizer
        self.sentence_dict = {}

    def __len__(self):
        return self.len

    def _load_datapair(self, path):
        """
        NOTE: For distributed sampling, this isn't equivalent to perfectly uniform sampling.
        In particular, each subset is perfectly represented in every batch! However, since we never
        repeat passes over the data, we never repeat any particular triple, and the split across
        nodes is random (since the underlying file is pre-shuffled), there's no concern here.
        """
        if torch.cuda.current_device() == 0:
            print(f"@@@ Loading datapair...")
        data = {"input": [], "output": [], "type": []}

        with open(path) as f:
            for line in f:
                source, target, type_ = line.strip().split('\t')
                source, target, type_ = int(source), int(target), int(type_)
                data["input"].append(source)
                data["output"].append(target)
                data["type"].append(type_)
        return data

    def _load_queries(self, path):
        if torch.cuda.current_device() == 0:
            print(f"@@@ Loading queries...")
        queries = {}

        with open(path) as f:
            for line in f:
                qid, query = line.strip().split('\t')
                qid = int(qid)
                queries[qid] = query
        return queries

    def _load_collection(self, path):
        if torch.cuda.current_device() == 0:
            print(f"@@@ Loading collection...")
        collection = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                try:
                    pid, passage, *_ = line.strip().split('\t')
                except:
                    if len(line.strip().split('\t')) == 1:
                        pid = line.strip()
                        passage = ""
                    else:
                        assert False
                assert pid == 'id' or int(pid) == line_idx
                collection.append(passage)
        return collection

    def _get_ids(self, s, max_length):
        ids = self.tokenizer.batch_encode_plus(
            [s],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return ids

    def convert_to_features(self, batch, idx):
        input_ = batch["input"]
        output_ = batch["output"]
        type_ = batch["type"]
        qid = copy.deepcopy(input_)
        pid = copy.deepcopy(output_)

        if type_ == 0: # query -> passage
            input_ = self.queries[input_]
            output_ = self.collection[output_]
        elif type_ == 1: # passage -> query
            input_ = self.collection[input_]
            output_ = self.queries[output_]
        else:
            raise NotImplementedError(f"Inappropriate qrels type: {type_}")
            
        if input_ not in self.sentence_dict.keys():
            source = self._get_ids(input_, max_length=self.hparams.max_input_length)
            self.sentence_dict[input_] = source
        else:
            source = self.sentence_dict[input_]

        if output_ not in self.sentence_dict.keys():
            target = self._get_ids(output_, max_length=self.hparams.max_output_length)
            self.sentence_dict[output_] = target
        else:
            target = self.sentence_dict[output_]

        assert (
            len(source["input_ids"].squeeze()) == self.hparams.max_input_length
            and len(target["input_ids"].squeeze()) == self.hparams.max_output_length
        ), print(f"length of source: {len(source['input_ids'])}\nlength of attention:  {len(target['input_ids'])}")


        if idx == 0 and torch.cuda.current_device() == 0:
            print(f"=" * 80)
            print(f"input: {input_}")
            print(f"output: {output_}")
            print(f"source: {source}")
            print(f"target: {target}")
            print(f"=" * 80)

        return source, target, input_, output_, qid, pid

    def __getitem__(self, idx):
        source, target, input_, output_, qid, pid = self.convert_to_features(
            self.dataset.iloc[idx], idx
        )
        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "target_ids": target_ids,
            "source_mask": src_mask,
            "target_mask": target_mask,
            "input": input_,
            "output": output_,
            "qid": qid,
            "pid": pid,
        }


class GENRETriplesDataset(Dataset):
    def __init__(self, tokenizer, split, hparams):
        self.hparams = hparams
        self.split = split
        if split == "train":
            data_path = self.hparams.train_file
        elif split == "validation":
            data_path = self.hparams.dev_file
        elif split == "test":
            data_path = self.hparams.test_file
        else:
            raise NotImplementedError(f"Inappropriate split type: {split}")

        self.queries = self._load_queries(os.path.join(self.hparams.dataset, self.hparams.queries))
        self.collection = self._load_collection(os.path.join(self.hparams.dataset, self.hparams.collection))

        self.dataset = self._load_triples(os.path.join(self.hparams.dataset, data_path))
        self.len = len(self.dataset)

        if torch.cuda.current_device() == 0:
            print(
                f"@@@ Loading from {os.path.join(self.hparams.dataset, data_path)}: {self.len}"
            )

        self.tokenizer = tokenizer
        self.sentence_dict = {}

    def __len__(self):
        return self.len

    def _load_datapair(self, path):
        """
        NOTE: For distributed sampling, this isn't equivalent to perfectly uniform sampling.
        In particular, each subset is perfectly represented in every batch! However, since we never
        repeat passes over the data, we never repeat any particular triple, and the split across
        nodes is random (since the underlying file is pre-shuffled), there's no concern here.
        """
        if torch.cuda.current_device() == 0:
            print(f"@@@ Loading datapair...")
        data = {"input": [], "output": [], "type": []}

        with open(path) as f:
            for line in f:
                source, target, type_ = line.strip().split('\t')
                source, target, type_ = int(source), int(target), int(type_)
                data["input"].append(source)
                data["output"].append(target)
                data["type"].append(type_)
        return data

    def _load_triples(self, path):
        """
        NOTE: For distributed sampling, this isn't equivalent to perfectly uniform sampling.
        In particular, each subset is perfectly represented in every batch! However, since we never
        repeat passes over the data, we never repeat any particular triple, and the split across
        nodes is random (since the underlying file is pre-shuffled), there's no concern here.
        """
        if torch.cuda.current_device() == 0:
            print("@@@ Loading triples...")

        triples = []
        with open(path) as f:
            for line_idx, line in enumerate(f):
                qid, pos, neg = ujson.loads(line)
                triples.append((qid, pos, neg))

        return triples

    def _load_queries(self, path):
        if torch.cuda.current_device() == 0:
            print(f"@@@ Loading queries...")
        queries = {}

        with open(path) as f:
            for line in f:
                qid, query = line.strip().split('\t')
                qid = int(qid)
                queries[qid] = query
        return queries

    def _load_collection(self, path):
        if torch.cuda.current_device() == 0:
            print(f"@@@ Loading collection...")
        collection = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                try:
                    pid, passage, *_ = line.strip().split('\t')
                except:
                    if len(line.strip().split('\t')) == 1:
                        pid = line.strip()
                        passage = ""
                    else:
                        assert False
                assert pid == 'id' or int(pid) == line_idx
                collection.append(passage)
        return collection

    def _get_ids(self, s, max_length):
        ids = self.tokenizer.batch_encode_plus(
            [s],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return ids

    def convert_to_features(self, input_, output_, type_, idx):
        #input_ = batch["input"]
        #output_ = batch["output"]
        #type_ = batch["type"]

        if type_ == 0: # query -> passage
            input_ = self.queries[input_]
            output_ = self.collection[output_]
        elif type_ == 1: # passage -> query
            input_ = self.collection[input_]
            output_ = self.queries[output_]
        else:
            raise NotImplementedError(f"Inappropriate qrels type: {type_}")
            
        if input_ not in self.sentence_dict.keys():
            source = self._get_ids(input_, max_length=self.hparams.max_input_length)
            self.sentence_dict[input_] = source
        else:
            source = self.sentence_dict[input_]

        if output_ not in self.sentence_dict.keys():
            target = self._get_ids(output_, max_length=self.hparams.max_output_length)
            self.sentence_dict[output_] = target
        else:
            target = self.sentence_dict[output_]

        assert (
            len(source["input_ids"].squeeze()) == self.hparams.max_input_length
            and len(target["input_ids"].squeeze()) == self.hparams.max_output_length
        ), print(f"length of source: {len(source['input_ids'])}\nlength of attention:  {len(target['input_ids'])}")


        if idx == 0 and torch.cuda.current_device() == 0:
            print(f"=" * 80)
            print(f"input: {input_}")
            print(f"output: {output_}")
            print(f"source: {source}")
            print(f"target: {target}")
            print(f"=" * 80)

        return source, target, input_, output_


    def random_sample(self, pids, k=1):
        return random.sample(pids, k=k)

    def __getitem__(self, idx):
        qid, pos_pids, neg_pids = self.dataset[idx]
        pos = self.random_sample(pos_pids)[0]
        neg = self.random_sample(neg_pids)[0]

        source, pos_target, input_, pos_output_ = self.convert_to_features(qid, pos, 0, idx)
        source, neg_target, input_, neg_output_ = self.convert_to_features(qid, neg, 0, idx)



        source_ids = source["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()

        pos_target_ids = pos_target["input_ids"].squeeze()
        pos_target_mask = pos_target["attention_mask"].squeeze()

        neg_target_ids = neg_target["input_ids"].squeeze()
        neg_target_mask = neg_target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids,
            "pos_target_ids": pos_target_ids,
            "neg_target_ids": neg_target_ids,

            "source_mask": src_mask,
            "pos_target_mask": pos_target_mask,
            "neg_target_mask": neg_target_mask,

            "input": input_,
            "pos_output_": pos_output_,
            "neg_output_": neg_output_,
            
            "qid": qid,
            "pos": pos,
            "neg": neg,
        }


if __name__ == "__main__":
    import argparse
    from utils.kobert_tokenizer import KoBERTTokenizer

    hparam_dict = dict(
        max_input_length=384,
        max_query_length=64,
        doc_stride=128,
        dataset="../dataset/korquadv1",
        train_file="KorQuAD_v1.0_train.json",
    )
    hparam = argparse.Namespace(**hparam_dict)
    
    tokenizer = KoBERTTokenizer.from_pretrained(
                    "skt/kobert-base-v1"
                )

    dataset = load_korquad_dataset(tokenizer, "train", hparam)