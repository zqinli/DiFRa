import json
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import math

from ..prompt.prompt import system_prompt, user_prompt


def _read_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{p} must be list[object]")
    return data


def _read_jsonl(p: Path):
    out = []
    with p.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def _read_any(p: Path):
    if p.suffix == ".jsonl":
        return _read_jsonl(p)
    if p.suffix == ".json":
        return _read_json(p)
    raise ValueError(f"unsupported file {p}")


def _strip_prefix(x: str) -> str:
    return x[3:] if isinstance(x, str) and x.startswith("ex:") else x


def _only_triples(triples):
    out = []
    for t in triples or []:
        h = _strip_prefix(str(t.get("h") or t.get("subject") or "").strip())
        r = _strip_prefix(str(t.get("r") or t.get("predicate") or "").strip())
        o = _strip_prefix(str(t.get("t") or t.get("object") or "").strip())
        if h and r and o:
            out.append({"h": h, "r": r, "t": o})
    return out


def _triples_to_text(triples):
    return "\n".join(f"{t['h']} {t['r']} {t['t']}" for t in triples) if triples else ""


def  _get_unk_token_stats(
    train_data_path,
    test_data_path, 
    bert_tokenizer,
):
    train_examples = _read_any(Path(train_data_path))
    test_examples = _read_any(Path(test_data_path))
    unk_tokens = []

    for train_ex in train_examples:
        keywords_list = train_ex.get("keywords") or train_ex.get("keywords") or []
        topics_list   = train_ex.get("topics") or train_ex.get("topic") or []

        all_concepts_set = set(keywords_list) | set(topics_list)
        all_concepts_list_sorted = sorted(list(all_concepts_set))
        concepts_str = "".join(all_concepts_list_sorted)

        unk_tokens.append(len(bert_tokenizer.tokenize(concepts_str)))

    for test_ex in test_examples:
        keywords_list = test_ex.get("keywords") or test_ex.get("keywords") or []
        topics_list   = test_ex.get("topics") or test_ex.get("topic") or []

        all_concepts_set = set(keywords_list) | set(topics_list)
        all_concepts_list_sorted = sorted(list(all_concepts_set))
        concepts_str = "".join(all_concepts_list_sorted)

        unk_tokens.append(len(bert_tokenizer.tokenize(concepts_str)))

    mean_unk_token = math.floor(sum(unk_tokens) / len(unk_tokens))
    max_unk_token = max(unk_tokens)

    return mean_unk_token, max_unk_token

def get_instruction_dataset(
    data_path: str | Path,
    max_qa_pair: int,
    infer_max_qa_pair: int,
    system_prompt: str,
    prompt_template: str,
    *,
    use_concepts: bool = False,
    num_concepts: Optional[int] = None,
    use_diffusion: bool = False,
    unk_token: str = "<unk>",
    use_knowledge_graph: bool = False,
    is_train: bool = False,
    bert_tokenizer = None,
) -> List[Dict[str, Any]]:

    examples = _read_any(Path(data_path))
    out: List[Dict[str, Any]] = []

    for ex in examples:
        system_prompt_template = system_prompt

        context       = (ex.get("context") or "").strip()
        qas_raw       = ex.get("qa") or ex.get("qas") or []
        triples       = _only_triples(ex.get("open_model_refiment_triples") or [])
        keywords_list = ex.get("keywords") or ex.get("keywords") or []
        topics_list   = ex.get("topics") or ex.get("topic") or []

        qas = []
        for qa in qas_raw:
            q_ = (qa.get("q") or qa.get("question") or "").strip()
            a_ = (qa.get("a") or qa.get("answer") or "").strip()
            if q_:
                qas.append({"question": q_, "answer": a_})
        
        if not qas:
            continue

        label = json.dumps(qas, ensure_ascii=False, separators=(',', ':'))

        all_concepts_set = set(keywords_list) | set(topics_list)
        all_concepts_list_sorted = sorted(list(all_concepts_set))
        concepts_str = " ".join(all_concepts_list_sorted)

        if use_concepts and use_diffusion:
            kw_generalized = " ".join([unk_token] * len(bert_tokenizer.tokenize(concepts_str)))

        kgs_text = _triples_to_text(triples) if use_knowledge_graph else ""

        filled_user_prompt = prompt_template.format(
            given_context=("Context: " + context),
            given_concepts=("Concepts: " + kw_generalized) if use_concepts and use_diffusion else "",
            given_kgs=("Knowledge graph: " + kgs_text) if use_knowledge_graph else "",
        )
        example_data = [
            {
                "question": "Question 1",
                "answer": "Answer 1"
            },
            {
                "question": "Question 2",
                "answer": "Answer 2"
            }
        ]
        example_json_string = json.dumps(example_data, indent=2)
        system_prompt_filled = system_prompt_template.format(
            num_qa_pairs=len(qas) if is_train else infer_max_qa_pair,
            output_example = example_json_string
        )

        out.append({
            "system_prompt": system_prompt_filled.strip(),
            "user_prompt":   filled_user_prompt.strip(),
            "label":         label,
            "concepts":      concepts_str.strip(),
            "cond":          context,
            "context":       context,
        })
    
    return out


def load_dataset(
    input_path: str,
    max_qa_pair: int,
    infer_max_qa_pair: int,
    *,
    use_concepts: bool = False,
    num_concepts: Optional[int] = None,
    use_diffusion: bool = False,
    unk_token: str = "<unk>",
    use_knowledge_graph: bool = False,
    bert_tokenizer = None,
):
    train_data_path = input_path['train_data_path']
    test_data_path = input_path['test_data_path']

    dataset_train = get_instruction_dataset(
        data_path=train_data_path,
        max_qa_pair=max_qa_pair,
        infer_max_qa_pair=infer_max_qa_pair,
        system_prompt=system_prompt,
        prompt_template=user_prompt,
        use_concepts=use_concepts,
        num_concepts=num_concepts,
        use_diffusion=use_diffusion,
        unk_token=unk_token,
        use_knowledge_graph=use_knowledge_graph,
        is_train=True,
        bert_tokenizer=bert_tokenizer,
    )
    
    dataset_test = get_instruction_dataset(
        data_path=test_data_path,
        max_qa_pair=max_qa_pair,
        infer_max_qa_pair=infer_max_qa_pair,
        system_prompt=system_prompt,
        prompt_template=user_prompt,
        use_concepts=use_concepts,
        num_concepts=num_concepts,
        use_diffusion=use_diffusion,
        unk_token=unk_token,
        use_knowledge_graph=use_knowledge_graph,
        bert_tokenizer=bert_tokenizer,
    )

    return dataset_train, dataset_test

