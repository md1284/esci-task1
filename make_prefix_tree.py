import os
import sys
import argparse 
import pickle
import pandas as pd
import numpy as np

from argparse import ArgumentParser
from transformers import T5Tokenizer
from tqdm import tqdm

def pickle_dump(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)
        
def pickle_load(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def load_collection(path):
    collection = []
    
    with open(path) as f:
        for line_idx, line in enumerate(f):
            pid, passage, *_ = line.strip().split('\t')
            assert pid == 'id' or int(pid) == line_idx
            collection.append(passage)
    return collection

def get_ids(tokenizer, s):
    ids = tokenizer.batch_encode_plus(
        [s],
        return_tensors="pt",
    )
    return ids

def add_subtrie(trie, ids):
    cur = trie
    for _id in ids:
        if _id not in cur:
            cur[_id] = {}
        cur = cur[_id]

def construct_prefix_tree(qid_to_candidate_pid, collection, tokenizer):
    qid_to_trie = {}

    length = []
    for qid, pids in tqdm(qid_to_candidate_pid.items()):
        assert qid not in qid_to_trie, qid

        trie = {}
        for pid in pids:
            psg = collection[pid]
            input_ids = [0] + get_ids(tokenizer, psg)["input_ids"].squeeze().tolist()
            length.append(len(input_ids))
            add_subtrie(trie, input_ids)
        qid_to_trie[qid] = trie

    print(np.min(length), np.max(length), np.median(length), np.mean(length))
    return qid_to_trie

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default=None, required=True, type=str)
    parser.add_argument("--collection", default=None, required=True, type=str)
    parser.add_argument("--qid_to_candidate_pid", default=None, required=True, type=str)
    parser.add_argument("--model_name_or_path", default=None, required=True, type=str)
    parser.add_argument("--save_path", default=None, required=True, type=str)
    args = parser.parse_args()

    collection = load_collection(os.path.join(args.dataset, args.collection))
    qid_to_candidate_pid = pickle_load(os.path.join(args.dataset, args.qid_to_candidate_pid))
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)

    qid_to_trie = construct_prefix_tree(qid_to_candidate_pid, collection, tokenizer)
    #pickle_dump(qid_to_trie, args.save_path)