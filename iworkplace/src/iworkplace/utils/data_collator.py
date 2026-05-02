from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
import torch.nn.utils.rnn as rnn_utils
import numpy as np

@dataclass
class QAGDataCollator(DataCollatorForLanguageModeling):

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        
        custom_keys = ['x_input_ids', 'x_input_mask', 'x_input_attention_mask']
        
        llm_examples = [] 
        custom_batches = {k: [] for k in custom_keys if k in examples[0]}
        
        for ex in examples:
            llm_ex_data = {}
            for k, v in ex.items():
                if k in custom_keys:
                    if k in custom_batches:
                        custom_batches[k].append(v)
                else:
                    llm_ex_data[k] = v
            llm_examples.append(llm_ex_data)

        batch = super().torch_call(llm_examples) 

        for key in custom_keys:
            if key in custom_batches:
                values = custom_batches[key]
                try:
                    if isinstance(values[0], torch.Tensor):
                       batch[key] = torch.stack(values)
                    else:
                       batch[key] = torch.tensor(values, dtype=torch.int64)
                except Exception as e:
                    print(f"!!!!!!!! ERROR: Failed to convert '{key}' to tensor !!!!!!!!")
                    print(f"Exception: {e}")
                    try:
                        print(f"Length of values[0]: {len(values[0])}")
                        if len(values) > 1:
                            print(f"Length of values[1]: {len(values[1])}")
                    except:
                        pass
                    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    raise e

        return batch