import json
import re
import ast
from typing import List, Dict, Any, Optional, Tuple
from vllm import LLM
from transformers import AutoTokenizer
from string import Template

FACT_SYSTEM_PROMPT = (
    "You are a Factual Consistency Evaluator. \n"
    "Your goal is to determine the factual fidelity of Candidate QA pairs and their relevance to the provided Reference QA pairs, using the Context and General Knowledge as supporting evidence.\n\n"
    "### Scoring Rubric (1-5)\n\n"
    "Score 5: Correct and Relevant\n"
    "- Factuality: The Candidate QA pair as a single unit is factually flawless and explicitly supported by the Context.\n"
    "- Alignment: The pair mirrors a 'Twin' in the Reference set. The Question's intent and the Answer's core information must be SEMANTICALLY EQUIVALENT to a specific Reference QA pair.\n"
    "- The 'Noise' Penalty: If the Candidate is factually true but includes extra \"fluff\" or describes a fact from the Context that the Reference ignored, it MUST be downgraded to Score 4.\n\n"
    "Score 4: Correct but Irrelevant\n"
    "- Factuality: The Candidate QA pair is empirically verifiable.\n"
    "- Divergence: If the Candidate's question has NO semantically equivalent match in the Reference QA list, it is considered Irrelevant, regardless of its factual accuracy. It is a 'correct answer to the wrong question.'\n\n"
    "Score 3: Partially Correct\n"
    "- Incompleteness: The pair captures the core intent but omits critical constraints or nuances found in the Context.\n"
    "- Minor Imprecision: The answer is 'directionally' correct but contains slight errors in details (dates, figures) while grounded in Context.\n\n"
    "Score 2: Unverifiable\n"
    "- Evidence Void: The information is absent from the Context.\n\n"
    "Score 1: Direct Contradiction\n"
    "- Conflict: The Candidate QA pair contradicts the Context.\n\n"
    "### Constraints\n"
    "- Output Format: MUST be a raw JSON list of objects.\n"
    "- Structure: [{\"index\": <int>, \"score\": <int>}]\n"
    "- NO markdown, NO explanations, NO deviations.\n"
    "- NO POSTSCRIPT: The response must start with '[' and end with ']'. Strictly no text after the JSON array.\n\n"
)

FACT_USER_TPL = Template(
    "Context:\n$context\n\n"
    "Reference QA:\n$ref_block\n\n"
    "Candidate QA:\n$qa_block\n\n"
)

def _one_shot_prompt(
    tokenizer: AutoTokenizer, 
    context: str, 
    candidate_pairs: List[Tuple[str, str]], 
    reference_pairs: List[Tuple[str, str]] = None
) -> str:
    if reference_pairs:
        ref_block = "\n".join(
            f"{i}. Question: {q}\n   Answer: {a}" for i, (q, a) in enumerate(reference_pairs)
        )
    else:
        ref_block = "None provided."

    qa_block = "\n".join(
        f"{i}. Question: {q}\n   Answer: {a}" for i, (q, a) in enumerate(candidate_pairs)
    )
    
    user_text = FACT_USER_TPL.substitute(
        context=context, 
        ref_block=ref_block, 
        qa_block=qa_block
    )
    
    messages = [
        {"role": "system", "content": FACT_SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class ConsistencyAssessor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        llm_instance: Optional[LLM] = None,
        tokenizer_instance: Optional[AutoTokenizer] = None,
        *,
        dtype: str = "bfloat16",
        gpu_memory_utilization: float = 0.9,
        tensor_parallel_size: int = 1,
        max_model_len: Optional[int] = None,
        stop: Optional[List[str]] = None,
        max_context_tokens: Optional[int] = None,
        one_shot_max_questions: int = 200,
        tokens_per_item_budget: int = 64,
        **vllm_kwargs,
    ):
        self.max_context_tokens = max_context_tokens
        self.one_shot_max_questions = int(one_shot_max_questions)
        self.tokens_per_item_budget = int(tokens_per_item_budget)

        if llm_instance and tokenizer_instance:
            self.model = llm_instance
            self.tokenizer = tokenizer_instance
        elif model_path:
            eng = dict(
                dtype=dtype,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
            )
            if max_model_len is not None:
                eng["max_model_len"] = max_model_len
            eng.update(vllm_kwargs)
            self.model = LLM(model=model_path, trust_remote_code=True, **eng)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )
        else:
            raise ValueError("Must provide either (llm_instance, tokenizer_instance) or a model_path.")
        
        self.base_sampling_params = dict(temperature=0.0, stop=["]", "Explanation:"],)

    def truncate_context(self, context: str) -> str:
        if not self.max_context_tokens:
            return context
        ids = self.tokenizer.encode(context, add_special_tokens=False)
        if len(ids) <= self.max_context_tokens:
            return context
        ids = ids[: self.max_context_tokens]
        return self.tokenizer.decode(ids, skip_special_tokens=True)

    @staticmethod
    def _strip_code_fences(t: str) -> str:
        t = t.strip()
        t = re.sub(r"^\s*```[a-zA-Z0-9_-]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t)
        return t.strip()

    @staticmethod
    def _find_top_level_array(t: str) -> Optional[str]:
        start = None
        depth = 0
        for i, ch in enumerate(t):
            if ch == "[":
                if depth == 0 and start is None:
                    start = i
                depth += 1
            elif ch == "]":
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start is not None:
                        return t[start : i + 1]
        return None

    @staticmethod
    def _remove_trailing_commas(s: str) -> str:
        return re.sub(r",\s*([}\]])", r"\1", s)

    @staticmethod
    def _quote_unquoted_keys(s: str) -> str:
        s = re.sub(r'([{\[,]\s*)([A-Za-z_][A-Za-z0-9_\-]*)\s*:', r'\1"\2":', s)
        return s

    @staticmethod
    def _single_to_double_quotes(s: str) -> str:
        s = re.sub(r"\'(index|score|label)\'\s*:", r'"\1":', s, flags=re.IGNORECASE)
        return s

    @staticmethod
    def _swap_bools_none_for_json(s: str) -> str:
        s = re.sub(r"\bTrue\b", "true", s)
        s = re.sub(r"\bFalse\b", "false", s)
        s = re.sub(r"\bNone\b", "null", s)
        return s

    @staticmethod
    def _swap_bools_none_for_python(s: str) -> str:
        s = re.sub(r"\btrue\b", "True", s)
        s = re.sub(r"\bfalse\b", "False", s)
        s = re.sub(r"\bnull\b", "None", s)
        return s

    def _lenient_json_list_parse(self, text: str) -> Optional[List[Any]]:
        if not text: return None
        t = self._strip_code_fences(text)
        arr = self._find_top_level_array(t)
        if not arr: return None

        try:
            data = json.loads(arr)
            return data if isinstance(data, list) else None
        except Exception: 
            pass

        s = arr
        s = self._remove_trailing_commas(s)
        s = self._quote_unquoted_keys(s)
        s = self._single_to_double_quotes(s)
        s_json = self._swap_bools_none_for_json(s)

        try:
            data = json.loads(s_json)
            if isinstance(data, list): return data
        except Exception: 
            pass

        s_py = self._swap_bools_none_for_python(s)
        try:
            data = ast.literal_eval(s_py)
            return data if isinstance(data, list) else None
        except Exception:
            return None

    @staticmethod
    def _normalize_item(obj: Dict[str, Any]) -> Dict[str, Any]:
        idx = obj.get("index", None)
        try:
            idx = int(idx) if idx is not None else None
        except: 
            idx = None
        
        raw_score = obj.get("score")
        final_score = 3
    
        if raw_score is not None:
            try:
                val = int(raw_score)
                if 1 <= val <= 5: 
                    final_score = val
                    return {"index": idx, "score": final_score}
            except: 
                pass

            if isinstance(raw_score, str):
                match = re.search(r'([1-5])', raw_score)
                if match:
                    final_score = int(match.group(1))
                    return {"index": idx, "score": final_score}

        old_label = str(obj.get("label", "")).lower()
        if "aligned" in old_label or "match" in old_label: 
            final_score = 5
        elif "support" in old_label: 
            final_score = 4
        elif "contra" in old_label: 
            final_score = 1
            
        return {"index": idx, "score": final_score}

    @staticmethod
    def _score_from_val(score_val: int) -> Tuple[int, float]:
        confidence_map = {
            5: 1.0, 4: 0.8, 3: 0.5, 2: 0.2, 1: 0.0
        }
        return score_val, confidence_map.get(score_val, 0.5)
    
    def parse_and_score_batch(self, llm_output_text: str, qa_pairs: List[Tuple[str, str]]) -> List[Dict[str, Any]]:
        if not qa_pairs: return []
            
        raw = (llm_output_text or "").strip()
        data: Optional[List[Any]] = self._lenient_json_list_parse(raw)

        if data is None:
            data = [{"index": i, "score": 3} for i in range(len(qa_pairs))]

        normalized = [self._normalize_item(obj) for obj in data if isinstance(obj, dict)]
        
        by_index: Dict[int, Dict[str, Any]] = {}
        seq_only: List[Dict[str, Any]] = []
        for item in normalized:
            if isinstance(item.get("index"), int):
                by_index[item["index"]] = item
            else:
                seq_only.append(item)

        results: List[Dict[str, Any]] = []
        seq_cursor = 0
        for i, (q, a) in enumerate(qa_pairs):
            item = by_index.get(i)
            if item is None:
                item = seq_only[seq_cursor] if seq_cursor < len(seq_only) else {"score": 3}
                seq_cursor += 1
            
            score_val = item.get("score", 3)
            final_score, confidence = self._score_from_val(score_val)

            results.append({
                "index": i,
                "original_question": q,
                "original_answer": a,
                "reference": { "score": score_val }, 
                "scores": {
                    "consistency_score": final_score,
                    "factuality_continuous": confidence
                },
                "raw_llm_output": raw if i == 0 else None
            })
        return results