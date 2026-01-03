import os
import sys
import json
import argparse
from transformers import set_seed

from run_evaluation import run_evaluation
from run_evaluation_tradition import run_evaluation_tradition
from src.utils.normalize_data import safe_parse_json

def load_dataset(file_path, max_prediction=10):
    with open(file_path, 'r', encoding='utf-8') as f:
        data_merged = []
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            context = data.get("context", "")
            label = data.get("label", "")
            prediction = data.get("prediction", "")
            data_merged.append({
                "context": context,
                "label": safe_parse_json(label),
                "prediction": safe_parse_json(prediction, max_pairs=max_prediction)
            })
    return data_merged

def main(
    input_file,
    llm_path,
    embedding_path,
    bert_model_path,
    eval_mode,
    ppl_model_path=None,
    gpu_mem_util=0.7,
    max_len=8192,
    lang="en",
    batch_size=16,
    max_qa_pairs=10,
    **kwargs,
):
    data = load_dataset(input_file, max_prediction=max_qa_pairs)
    
    dir_path, full_filename = os.path.split(input_file)
    file_name, file_ext = os.path.splitext(full_filename)
    output_dir = os.path.join(dir_path, file_name)
    output_dir = os.path.join(output_dir, os.path.basename(llm_path))

    os.makedirs(output_dir, exist_ok=True)
    
    if eval_mode == "llm":
        out_path = os.path.join(output_dir, "evaluation.json")
        run_evaluation(
            data=data,
            output_filename=out_path,
            llm_path=llm_path,
            embedding_path=embedding_path,
            bert_model_path=bert_model_path,
            gpu_mem_util=gpu_mem_util,
            max_len=max_len,
            num_gpus=kwargs.get("num_gpus", 1)
        )
        
    elif eval_mode == "tradition":
        out_path_tradition = os.path.join(output_dir, "evaluation_tradition.json")
        run_evaluation_tradition(
            data=data,
            output_filename=out_path_tradition,
            ppl_model_path=ppl_model_path,
            bert_model_path=bert_model_path,
            lang=lang,
            batch_size=batch_size
        )
        
    elif eval_mode == "both":
        out_path = os.path.join(output_dir, "evaluation.json")
        run_evaluation(
            data=data,
            output_filename=out_path,
            llm_path=llm_path,
            embedding_path=embedding_path,
            bert_model_path=bert_model_path,
            gpu_mem_util=gpu_mem_util,
            max_len=max_len,
            num_gpus=kwargs.get("num_gpus", 1)
        )
        
        out_path_tradition = os.path.join(output_dir, "evaluation_tradition.json")
        run_evaluation_tradition(
            data=data,
            output_filename=out_path_tradition,
            ppl_model_path=ppl_model_path,
            bert_model_path=bert_model_path,
            lang=lang,
            batch_size=batch_size
        )
        
    else:
        sys.exit(1)

    sys.exit(0)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="runner for evaluation")
    p.add_argument("--input_file", required=True, help="Path to input data file")
    p.add_argument("--llm_path", required=True, help="Path to LLM model")
    p.add_argument("--embedding_path", required=True, help="Path to embedding model")
    p.add_argument("--bert_model_path", required=True, help="Path to BERT model")
    p.add_argument("--ppl_model_path", default=None, help="Path to PPL model (for traditional evaluation)")
    p.add_argument("--gpu_mem_util", type=float, default=0.7, help="GPU memory utilization ratio")
    p.add_argument("--max_len", type=int, default=8192, help="Maximum sequence length")
    p.add_argument("--lang", type=str, default="en", help="Language type (default: en)")
    p.add_argument("--batch_size", type=int, default=1024, help="Batch size for traditional evaluation")
    p.add_argument("--max_qa_pairs", type=int, default=10, help="Maximum number of QA pairs")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--num_gpus", type=int, default=1, help="Number of GPUs to use")
    
    p.add_argument(
        "--eval_mode", 
        required=True, 
        choices=["llm", "tradition", "both"],
        help="Evaluation mode: 'modern' (run run_evaluation), 'traditional' (run run_evaluation_tradition), or 'both' (run both)"
    )

    args = p.parse_args()
    set_seed(args.seed)
    main(**vars(args))