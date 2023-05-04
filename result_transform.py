import os
import numpy as np

hypothesis_name = "ddp_neg_1dir_q2d_vocab_cbmlul_swl.results"
outputs_path = "outputs/ddp_neg_1dir_q2d_vocab_cbmlul_swl"

source_path = os.path.join(outputs_path, hypothesis_name)
target_path = os.path.join("hypothesis", hypothesis_name)


query_id_to_sorted_list = {}
with open(source_path, "r") as f:
    for line in f:
        query_id, _, product_id, loss, _ = line.split()
        
        if query_id not in query_id_to_sorted_list.keys():
            query_id_to_sorted_list[query_id] = []
        query_id_to_sorted_list[query_id].append([product_id, float(loss)])


min_trec_eval_score = 0
max_trec_eval_score = 128

with open(target_path, "w") as f:
    for query_id, sorted_list in query_id_to_sorted_list.items():
        sorted_list = sorted(sorted_list, key=lambda x: x[1])
        n = len(sorted_list)
        scores = list(np.arange(min_trec_eval_score, max_trec_eval_score, max_trec_eval_score / n).round(3)[::-1][:n])

        for rank, ((product_id, loss), score) in enumerate(zip(sorted_list, scores)):
            f.write(f"{query_id} Q0 {product_id} {rank} {score} {hypothesis_name}\n")
