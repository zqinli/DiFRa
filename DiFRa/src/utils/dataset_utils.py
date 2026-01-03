from datasets import Dataset
from typing import Dict, Literal, Union, List, Any
import torch

def build_cond_ids_mask(
    src: str,
    trg: str,
    bert_tokenizer,
    seq_len: int = 128,
) -> Dict[str, List[int]]:
    assert bert_tokenizer is not None

    pad_id = bert_tokenizer.pad_token_id
    cls_id = bert_tokenizer.cls_token_id
    sep_id = bert_tokenizer.sep_token_id
      
    src_ids = bert_tokenizer(src, add_special_tokens=False, max_length=seq_len, truncation=True)["input_ids"]
    trg_ids = bert_tokenizer(trg, add_special_tokens=False, max_length=seq_len, truncation=True)["input_ids"]
        
    if not src_ids or not trg_ids:
        raise ValueError("Tokenization resulted in empty sequences.")

    while len(src_ids) + len(trg_ids) > seq_len - 3:
        if len(src_ids) > len(trg_ids):
            src_ids.pop()
        elif len(trg_ids) > 0:
            trg_ids.pop()
        else:
            break
            
    merged = [cls_id] + src_ids + [sep_id] + trg_ids + [sep_id]
    
    noise_mask_core = [0] * (len(src_ids) + 2) + [1] * len(trg_ids) + [0]
    
    attn_mask_core = [1] * len(merged)
    
    pad_len = seq_len - len(merged)
    if pad_len < 0:
        merged = merged[:seq_len]
        x_input_mask = noise_mask_core[:seq_len]
        x_input_attention_mask = attn_mask_core[:seq_len]
    elif pad_len > 0:
        merged += [pad_id] * pad_len
        x_input_mask = noise_mask_core + [0] * pad_len
        x_input_attention_mask = attn_mask_core + [0] * pad_len
    else:
        x_input_mask = noise_mask_core
        x_input_attention_mask = attn_mask_core
        
    return {
        "input_ids": merged, 
        "x_input_mask": x_input_mask,
        "x_input_attention_mask": x_input_attention_mask
    }


def tokenize_dataset(
    dataset: Dataset,
    tokenizer,
    model_type,
    max_length: int = 4096,
    *,
    for_inference: bool = False,
    use_diffusion: bool = False,
    bert_tokenizer=None,
    bert_embed=None,
    cond_max_length: int = 512,
):
    if not hasattr(dataset, "map"):
        dataset = Dataset.from_list(list(dataset))

    if use_diffusion:
        assert bert_tokenizer is not None
        for special in ["pad_token_id", "sep_token_id", "eos_token_id"]:
            bt_id = getattr(bert_tokenizer, special, None)
            if bt_id is not None:
                assert isinstance(bt_id, int)

    def tokenize_fn(ex: Dict) -> Dict:
        if model_type == "instruct":
            tokenized = {
                "prompt": [
                    {"role": "system", "content": ex.get("system_prompt")},
                    {"role": "user", "content":  ex.get("user_prompt")}
                ],
                "completion": [
                    {"role": "assistant", "content": ex.get("label")}
                ]
            }
        elif model_type == "chat":
            tokenized = {
                "prompt": [
                    {"role": "user", "content":  f"{ex.get('system_prompt')}\n\n{ex.get('user_prompt')}"}
                ],
                "completion": [
                    {"role": "assistant", "content": ex.get("label")}
                ]
            }
        else :
            raise ValueError(f"Unknown model_type: '{model_type}'.")

        if use_diffusion:
            src = ex.get("cond", "")
            trg = ex.get("concepts", "")
            if src and trg:
                cond_pack = build_cond_ids_mask(
                    src=src, trg=trg,
                    bert_tokenizer=bert_tokenizer,
                    seq_len=cond_max_length,
                )
                tokenized["x_input_ids"] = cond_pack["input_ids"]
                tokenized["x_input_mask"] = cond_pack["x_input_mask"]
                tokenized["x_input_attention_mask"] = cond_pack["x_input_attention_mask"]
            else:
                raise ValueError("Input sample must contain 'cond' and 'concepts' fields when use_diffusion is True")
        return tokenized
    return dataset.map(tokenize_fn, desc=f"Tokenizing dataset for {model_type} models")