import re
import json
import math
from typing import List, Dict, Any, Tuple, Optional
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

SYSTEM_PROMPT = (
    "You are a careful semantic clustering assistant for QA pairs.\n"
    "Task: partition the given QA pairs into clusters such that items in the same cluster make the SAME semantic claim.\n"
    "\n"
    "STRICT OUTPUT CONTRACT:\n"
    "- Output ONLY a valid JSON array of integer lists (list of clusters), e.g. [[0,2,5],[1,4],[3]].\n"
    "- Do NOT wrap the whole output in quotes. No markdown fences. No explanations or extra text.\n"
    "- Indices must be integers referring ONLY to the indices shown in the user message (starting at 0). "
)

USER_PROMPT_TPL = (
    "Shared context:\n{context}\n\n"
    "QA pairs (each line starts with its index):\n{qa_block}\n\n"
    "Return ONLY the clusters in JSON (list of lists of indices)."
)

class SemanticEntropy:
    def __init__(
        self,
        model_path: Optional[str] = None,
        llm_instance: Optional[LLM] = None,
        tokenizer_instance: Optional[AutoTokenizer] = None,
        **vllm_kwargs,
    ):
        if llm_instance and tokenizer_instance:
            self.model, self.tokenizer = llm_instance, tokenizer_instance
        elif model_path:
            self.model = LLM(model=model_path, trust_remote_code=True, **vllm_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )
        else:
            raise ValueError("provide (llm_instance & tokenizer_instance) or model_path")
        
        self.gen_cfg = SamplingParams(temperature=0.0, max_tokens=512)

    @staticmethod
    def _entropy(clusters: List[List[int]], n: int) -> Dict[str, float]:
        if n <= 1:
            return dict(entropy=0.0, normalized_entropy=0.0)
        h = 0.0
        for c in clusters:
            p = len(c) / n
            h -= p * math.log2(p) if p else 0.0
        
        if n == 1:
            return dict(entropy=h, normalized_entropy=0.0)
        return dict(entropy=h, normalized_entropy=h / math.log2(n))

    def build_prompt(self, ctx: str, qa_pairs: List[Tuple[str, str]]) -> str:
        qa_block = "\n\n".join(
            f"=== QA Pair {i} ===\nQ: {q.strip()}\nA: {a.strip()}"
            for i, (q, a) in enumerate(qa_pairs)
        )

        user = USER_PROMPT_TPL.format(context=ctx, qa_block=qa_block)
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(msgs, tokenize=False,
                                                  add_generation_prompt=True)

    @staticmethod
    def _extract_json_array(text: str) -> str:
        t = text.strip()
        t = re.sub(r"^```[a-zA-Z]*", "", t)
        t = re.sub(r"```$", "", t)
        start = t.find('[')
        end   = t.rfind(']')
        if start == -1 or end == -1 or end <= start:
            raise ValueError("no outer JSON array found")
        return t[start:end + 1]

    def parse_and_score(self, llm_output_text: str, n_pairs: int) -> Dict[str, Any]:
        if n_pairs <= 1:
            return {"num_clusters": n_pairs, "entropy": 0.0, "normalized_entropy": 0.0}

        try:
            arr = self._extract_json_array(llm_output_text)
            clusters_raw = json.loads(arr)
            assert isinstance(clusters_raw, list)
            clusters = [sorted({int(i) for i in c})
                        for c in clusters_raw if isinstance(c, list)]
        except Exception:
            clusters = [list(range(n_pairs))]

        seen, clean = set(), []
        for c in clusters:
            c = [i for i in c if 0 <= i < n_pairs and i not in seen]
            if c:
                seen.update(c)
                clean.append(c)
        
        for i in range(n_pairs):
            if i not in seen:
                clean.append([i])
        
        if not clean and n_pairs > 0:
             clean = [[i] for i in range(n_pairs)]

        met = self._entropy(clean, n_pairs)
        return {
            "num_clusters": len(clean),
            "semantic_diversity_entropy": met["entropy"],
            "normalized_entropy": met["normalized_entropy"],
        }