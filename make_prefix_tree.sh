python make_prefix_tree.py \
--dataset {dataset_path} \
--collection {collection.tsv path} \
--qid_to_candidate_pid {qid_to_candidate_pid.pickle path} \
--model_name_or_path BeIR/query-gen-msmarco-t5-base-v1 \
--save_path {qid_to_trie.pickle path}