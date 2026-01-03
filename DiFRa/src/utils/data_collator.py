from dataclasses import dataclass
from typing import Any, Dict, List
import torch
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

@dataclass
class QAGDataCollator(DataCollatorForLanguageModeling):

    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        
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
                if isinstance(values[0], torch.Tensor):
                    batch[key] = torch.stack(values)
                else:
                    batch[key] = torch.tensor(values, dtype=torch.int64)

        return batch