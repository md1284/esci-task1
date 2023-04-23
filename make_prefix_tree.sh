python make_prefix_tree.py \
--dataset ../esci-data/shopping_queries_dataset \
--collection collection.tsv \
--qid_to_candidate_pid qid_to_candidate_pid.pickle \
--model_name_or_path BeIR/query-gen-msmarco-t5-base-v1 \
--save_path qid_to_trie.pickle