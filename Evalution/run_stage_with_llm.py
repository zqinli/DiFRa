import json
import gc
import torch
import numpy as np
import sys
import pickle
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from src.metrics.diversity.semantic_entropy import SemanticEntropy
from src.metrics.consistency.consistency_assessor import ConsistencyAssessor

from src.metrics.consistency.consistency_assessor import _one_shot_prompt as assessor_prompt_builder


def safe_mean(data_list: list) -> float:
    numeric_list = [x for x in data_list if isinstance(x, (int, float))]
    return float(np.mean(numeric_list)) if numeric_list else 0.0

def run_stage1(data, llm_path, output_path, gpu_mem_util=0.8, max_len=4096, num_gpus=1):
    final_gpu_mem_util = max(gpu_mem_util, 0.9)
    
    llm = LLM(
        model=llm_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=final_gpu_mem_util, 
        max_model_len=max_len,
        dtype="auto",
        seed=42
    )
    tokenizer = AutoTokenizer.from_pretrained(llm_path)

    semantic_entropy = SemanticEntropy(llm_instance=llm, tokenizer_instance=tokenizer)
    assessor = ConsistencyAssessor(llm_instance=llm, tokenizer_instance=tokenizer)

    se_prompts_to_run = []
    as_prompts_to_run = []
    
    as_params_list = []

    se_metadata = [] 
    as_metadata = [] 

    for i, item in enumerate(data):
        context = item["context"]
        pred_qa_pairs = [(qa["question"], qa["answer"]) for qa in item["prediction"]]

        raw_refs = item.get("label", [])
        if isinstance(raw_refs, str):
            try:
                raw_refs = json.loads(raw_refs)
            except:
                raw_refs = []
                
        ref_qa_pairs = [
            (qa["question"], qa.get("answer", qa.get("label", ""))) 
            for qa in raw_refs
        ]
        
        n_pairs = len(pred_qa_pairs)
        questions = [q for q, _ in pred_qa_pairs]

        se_metadata.append((i, n_pairs))
        if n_pairs > 1:
            se_prompts_to_run.append(semantic_entropy.build_prompt(context, pred_qa_pairs))
        
        as_metadata.append((i, pred_qa_pairs)) 
        if n_pairs > 0:
            if n_pairs > assessor.one_shot_max_questions:
                as_metadata[-1] = (i, None) 
            else:
                as_prompts_to_run.append(
                    assessor_prompt_builder(tokenizer, context, pred_qa_pairs, ref_qa_pairs)
                )
                
                current_max_tokens = max(256, assessor.tokens_per_item_budget * n_pairs + 64)
                
                as_params_list.append(
                    SamplingParams(max_tokens=current_max_tokens, **assessor.base_sampling_params, include_stop_str_in_output=True)
                )

    GENERAL_BATCH_SIZE = 4
    
    se_raw_results = []
    as_raw_results = []

    if se_prompts_to_run:
        for i in range(0, len(se_prompts_to_run), GENERAL_BATCH_SIZE):
            batch_prompts = se_prompts_to_run[i : i + GENERAL_BATCH_SIZE]
            outputs = llm.generate(batch_prompts, semantic_entropy.gen_cfg, use_tqdm=False) 
            se_raw_results.extend([out.outputs[0].text.strip() for out in outputs])

    if as_prompts_to_run:
        for i in range(0, len(as_prompts_to_run), GENERAL_BATCH_SIZE):
            batch_prompts = as_prompts_to_run[i : i + GENERAL_BATCH_SIZE]
            batch_params = as_params_list[i : i + GENERAL_BATCH_SIZE]
            outputs = llm.generate(batch_prompts, batch_params, use_tqdm=False)
            as_raw_results.extend([out.outputs[0].text.strip() for out in outputs])

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    se_iter = iter(se_raw_results)
    as_iter = iter(as_raw_results)

    semantic_entropy_results_llm = [None] * len(data)
    consistency_results_by_llm = [None] * len(data)

    for (i, n_pairs) in se_metadata:
        if n_pairs <= 1:
            se_result = {"num_clusters": n_pairs, "entropy": 0.0, "normalized_entropy": 0.0}
        else:
            raw_txt = next(se_iter)
            se_result = semantic_entropy.parse_and_score(raw_txt, n_pairs)
        semantic_entropy_results_llm[i] = se_result

    for (i, pred_qa_pairs) in as_metadata:
        if not pred_qa_pairs: 
            consistency_results_by_llm[i] = []
        else:
            raw_txt = next(as_iter)
            item_results = assessor.parse_and_score_batch(raw_txt, pred_qa_pairs)
            consistency_results_by_llm[i] = item_results

    all_llm_consistency_scores = [
        (it.get("consistency_score")
         if "consistency_score" in it
         else it.get("scores", {}).get("consistency_score"))
        for res_list in consistency_results_by_llm
        if res_list is not None
        for it in res_list
        if isinstance(it, dict)
    ]

    se_clusters = []
    se_entropy = []
    se_norm_entropy = []
    
    for res in semantic_entropy_results_llm:
        if res is not None:
            se_clusters.append(res.get('num_clusters', 0))
            se_entropy.append(res.get('semantic_diversity_entropy', 0))
            se_norm_entropy.append(res.get('normalized_entropy', 0))

    avg_metrics = {
        "semantic_entropy_llm_avg_clusters": safe_mean(se_clusters),
        "semantic_entropy_llm_avg_entropy": safe_mean(se_entropy),
        "semantic_entropy_llm_avg_normalized_entropy": safe_mean(se_norm_entropy),
        "consistency_llm_avg_score": safe_mean(all_llm_consistency_scores),
    }

    json.dump(avg_metrics, open(output_path, "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    
    sys.exit(0)

if __name__ == "__main__":
    data, args = pickle.load(sys.stdin.buffer)
    run_stage1(**args, data=data)