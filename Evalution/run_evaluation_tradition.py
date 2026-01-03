import json
import nltk
import torch
from typing import Any, Dict, List, Literal
from functools import lru_cache
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from bert_score import BERTScorer
from tqdm import tqdm

@lru_cache(maxsize=None)
def _tok(text: str) -> List[str]:
    return nltk.word_tokenize(text)

_SM = SmoothingFunction().method4
_BLEU_W = {
    1: (1.0, 0, 0, 0),
    2: (0.5, 0.5, 0, 0),
    3: (1/3, 1/3, 1/3, 0),
    4: (0.25, 0.25, 0.25, 0.25),
}

def compute_bleu_n(ref: str, cand: str, n: int) -> float:
    return sentence_bleu(
        [_tok(ref)], _tok(cand),
        weights=_BLEU_W[n],
        smoothing_function=_SM
    )

_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

def compute_rouge_max(refs: List[str], cand: str) -> Dict[str, float]:
    scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    for r in refs:
        s = _ROUGE.score(r, cand)
        scores = {k: max(scores[k], s[k].fmeasure) for k in scores}
    return scores

def compute_meteor_max(refs: List[str], cand: str) -> float:
    cand_tok = _tok(cand)
    return max(
        meteor_score([_tok(r)], cand_tok)
        for r in refs
    )

def compute_f1_max(refs: List[str], cand: str) -> float:
    cand_set = set(_tok(cand))
    best = 0.0
    for r in refs:
        ref_set = set(_tok(r))
        inter = len(cand_set & ref_set)
        if inter == 0:
            continue
        p = inter / len(cand_set)
        rcl = inter / len(ref_set)
        best = max(best, 2 * p * rcl / (p + rcl))
    return best

def compute_perplexity_batch(sentences: List[str],
                             model,
                             tokenizer,
                             device="cuda",
                             max_length=512) -> List[float]:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        if getattr(model, "config", None) is not None:
            model.config.pad_token_id = tokenizer.pad_token_id

    clean = [s for s in sentences if s.strip()]
    if not clean:
        return []

    enc = tokenizer(
        clean,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    ).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(**enc, labels=enc["input_ids"])
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = enc["input_ids"][:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            ignore_index=tokenizer.pad_token_id,
            reduction="none"
        )
        token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        ).view(shift_labels.size())

        valid = (shift_labels != tokenizer.pad_token_id).sum(dim=1).clamp(min=1)
        sent_loss = token_loss.sum(dim=1) / valid
        return torch.exp(sent_loss).tolist()

def evaluate(data: List[Dict[str, Any]],
             ppl_model_path: str,
             bert_model_path: str,
             lang: str = "en",
             batch_size: int = 1024,
             aggregate: Literal["per-cand", "max-per-item", "mean-per-item"] = "mean-per-item"
             ) -> Dict[str, float]:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ppl_tokenizer = AutoTokenizer.from_pretrained(ppl_model_path)
    ppl_model = AutoModelForCausalLM.from_pretrained(ppl_model_path).to(device)
    if getattr(ppl_model, "config", None) is not None and ppl_model.config.pad_token_id is None:
        ppl_model.config.pad_token_id = ppl_tokenizer.eos_token_id

    results: Dict[str, List[float]] = {
        "BLEU-1": [], "BLEU-2": [], "BLEU-3": [], "BLEU-4": [],
        "METEOR": [], "F1": [],
        "ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": [],
        "Perplexity": [], "BERTScore": [],
    }

    cfg = AutoConfig.from_pretrained(bert_model_path)
    
    scorer = BERTScorer(
        model_type=bert_model_path,
        num_layers=cfg.num_hidden_layers,
        device=device,
        batch_size=batch_size,
        lang=lang,
        rescale_with_baseline=False,
    )

    def _accumulate_per_item(name: str, metric_vals_for_preds: List[float]):
        if aggregate == "per-cand":
            results[name].extend(metric_vals_for_preds)
        elif aggregate == "max-per-item":
            results[name].append(max(metric_vals_for_preds) if metric_vals_for_preds else 0.0)
        elif aggregate == "mean-per-item":
            results[name].append(sum(metric_vals_for_preds) / len(metric_vals_for_preds) if metric_vals_for_preds else 0.0)

    for item in tqdm(data, desc="Evaluating", unit="item"):
        refs = [
            (ref.get("question", "") + " " + ref.get("answer", "")).strip()
            for ref in item.get("label", [])
        ]
        refs = [r for r in refs if r] or [""]

        preds = [
            (pred.get("question", "") + " " + pred.get("answer", "")).strip()
            for pred in item.get("prediction", [])
        ]
        preds = [p for p in preds if p]
        if not preds:
            continue

        per_pred_bleu1, per_pred_bleu2, per_pred_bleu3, per_pred_bleu4 = [], [], [], []
        per_pred_meteor, per_pred_f1 = [], []
        per_pred_r1, per_pred_r2, per_pred_rl = [], [], []

        for pred in preds:
            for n, bucket in zip((1,2,3,4),
                                 (per_pred_bleu1, per_pred_bleu2, per_pred_bleu3, per_pred_bleu4)):
                bucket.append(max(compute_bleu_n(r, pred, n) for r in refs))
            per_pred_meteor.append(compute_meteor_max(refs, pred))
            per_pred_f1.append(compute_f1_max(refs, pred))
            rg = compute_rouge_max(refs, pred)
            per_pred_r1.append(rg["rouge1"])
            per_pred_r2.append(rg["rouge2"])
            per_pred_rl.append(rg["rougeL"])

        for name, vals in [
            ("BLEU-1", per_pred_bleu1),
            ("BLEU-2", per_pred_bleu2),
            ("BLEU-3", per_pred_bleu3),
            ("BLEU-4", per_pred_bleu4),
            ("METEOR", per_pred_meteor),
            ("F1", per_pred_f1),
            ("ROUGE-1", per_pred_r1),
            ("ROUGE-2", per_pred_r2),
            ("ROUGE-L", per_pred_rl),
        ]:
            _accumulate_per_item(name, vals)

        ppl_vals = compute_perplexity_batch(preds, ppl_model, ppl_tokenizer, device=device)
        _accumulate_per_item("Perplexity", ppl_vals)

        per_pred_berts = []
        for pred in preds:
            hyp_list = [pred] * len(refs)
            P, R, F = scorer.score(hyp_list, refs)
            per_pred_berts.append(float(F.max().item()))
        
        _accumulate_per_item("BERTScore", per_pred_berts)

    return {k: (sum(v) / len(v) if v else 0.0) for k, v in results.items()}

def run_evaluation_tradition(data, output_filename, ppl_model_path, bert_model_path, lang, batch_size):
    metrics = evaluate(
        data,
        ppl_model_path=ppl_model_path,
        bert_model_path=bert_model_path,
        lang=lang,
        batch_size=batch_size
    )
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)