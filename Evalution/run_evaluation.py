import subprocess
import sys
import pickle
import tempfile
import json
import os
import time

def run_evaluation(data, output_filename, llm_path, embedding_path, bert_model_path,
                   gpu_mem_util=0.7, max_len=4096, num_gpus=1):
    stage1_out = output_filename.replace(".json", "_stage1.json")

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        pickle.dump((data, dict(llm_path=llm_path, output_path=stage1_out,
                                gpu_mem_util=gpu_mem_util, max_len=max_len, num_gpus=num_gpus)),
                    tmp)
        tmp.flush()

        result = subprocess.run(
            [sys.executable, "run_stage_with_llm.py"],
            stdin=open(tmp.name, "rb"),
            check=True
        )

        if result.returncode != 0:
            raise RuntimeError("Stage 1 failed: LLM subprocess exited with error.")
            
    os.unlink(tmp.name)
    time.sleep(10)