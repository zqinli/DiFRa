import torch
import numpy as np
import random
import os
import argparse
import json
from datetime import datetime
from typing import Dict, Optional
from trl import SFTConfig

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
from transformers import AutoTokenizer, EarlyStoppingCallback, set_seed
from torch.optim import AdamW

from src.dataset.load_dataset import load_dataset
from src.models.modeling_qag import QAGConfig, QAGForCausalLM
from src.utils.dataset_utils import tokenize_dataset
from src.utils.trainer import QAGTrainer
from src.utils.data_collator import QAGDataCollator

os.environ["TOKENIZERS_PARALLELISM"] = "false" 

class QAGRunner:
    UNK_TOKEN = "<unk>"
    UNK_TOKEN_ID = 0
    instruction_type = "instruct"

    def __init__(
        self,
        dataset_name: str,
        input_path: Dict[str, str],
        max_prompt_length: int,
        max_qa_pair: int,
        infer_max_qa_pair: int,
        model_type: str,
        model_name: str,
        load_model_accuracy: str,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        save_model: bool,
        output_dir: str,
        load_from_pretrained: bool,
        pretrained_model_name: Optional[str],
        freeze_llm: bool,
        use_lora: bool,
        use_lora_on_llama_model: bool,
        use_lora_on_denoiser: bool,
        use_training: bool,
        use_inference: bool,
        use_concepts: bool,
        num_concepts: Optional[int],
        use_diffusion: bool,
        diffusion_steps: int,
        bert_model_name: Optional[str],
        cond_max_length: Optional[int],
        use_ema: bool,
        lambda_diff: float,
        diffusion_mlp_block_num: int,
        use_knowledge_graph: bool,
        seed: int,
    ):
        locals_ = locals()
        locals_.pop("self")
        self.__dict__.update(locals_)
        QAGRunner.set_instruction_type(self.model_type)

    @classmethod
    def set_instruction_type(cls, instruction_type):
        cls.instruction_type = instruction_type

    def _load_tokenizer(self):
        tokenizer_src = (self.pretrained_model_name if self.load_from_pretrained and self.pretrained_model_name else self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.use_diffusion:
            self.tokenizer.add_special_tokens({"additional_special_tokens": [self.UNK_TOKEN]})
            self.UNK_TOKEN_ID = self.tokenizer.convert_tokens_to_ids(self.UNK_TOKEN)

        self.bert_tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name) if self.bert_model_name else None

    def _load_dataset(self):
        if self.dataset_name not in ["drop", "squad"]:
            raise ValueError("Only 'drop' and 'squad' datasets are supported for now.")
        self.dataset_train, self.dataset_test = load_dataset(
            input_path=self.input_path,
            max_qa_pair=self.max_qa_pair,
            infer_max_qa_pair=self.infer_max_qa_pair,
            use_concepts=self.use_concepts,
            num_concepts=self.num_concepts,
            use_diffusion=self.use_diffusion,
            unk_token=self.UNK_TOKEN,
            use_knowledge_graph=self.use_knowledge_graph,
            bert_tokenizer=self.bert_tokenizer,
        )

    def _prepare_lora(self):
        if not self.use_lora:
            return
        
        if self.use_lora_on_llama_model:
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                inference_mode=False,
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.CAUSAL_LM,
            )
            self.model.llama_model = get_peft_model(self.model.llama_model, lora_cfg)
            
        if self.use_lora_on_denoiser:
            lora_cfg_denoiser = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                inference_mode=False,
                target_modules=["query", "value"],
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            self.model.diffusion_model.denoiser = get_peft_model(
                self.model.diffusion_model.denoiser, 
                lora_cfg_denoiser
            )
            for name, param in self.model.diffusion_model.denoiser.named_parameters():
                if "time_embed" in name or "time_modulator" in name:
                    param.requires_grad = True

    def _load_model(self):
        if self.load_from_pretrained:
            cfg = QAGConfig.from_pretrained(self.pretrained_model_name)
            self.model = QAGForCausalLM.from_pretrained(self.pretrained_model_name, config=cfg, trust_remote_code=True)
        else:
            cfg = QAGConfig(
                model_name_or_path=self.model_name,
                load_model_accuracy=self.load_model_accuracy,
                freeze_llm=self.freeze_llm,
                use_diffusion=self.use_diffusion,
                bert_model_name_or_path=self.bert_model_name,
                num_concepts=self.num_concepts,
                unk_token=self.UNK_TOKEN,
                unk_token_id=self.UNK_TOKEN_ID,
                use_flash_att=False,
                use_ema=self.use_ema,
                lambda_diff=self.lambda_diff,
                diffusion_mlp_block_num=self.diffusion_mlp_block_num,
            )
            self.model = QAGForCausalLM(cfg)

        if self.tokenizer.pad_token_id is not None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
        else:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id

        vocab_size = len(self.tokenizer)
        self.model.llama_model.resize_token_embeddings(vocab_size)
        self.model.llama_model.config.vocab_size = vocab_size
        
        self.device = next(self.model.parameters()).device

    def start_training(self):
        dataset_train = tokenize_dataset(
            Dataset.from_list(self.dataset_train),
            self.tokenizer,
            model_type=QAGRunner.instruction_type,
            max_length=self.max_prompt_length,
            for_inference=False,
            use_diffusion=self.use_diffusion,
            bert_tokenizer=self.bert_tokenizer,
            cond_max_length=self.cond_max_length,
        )

        eval_dataset = tokenize_dataset(
            Dataset.from_list(self.dataset_test),
            self.tokenizer,
            model_type=QAGRunner.instruction_type,
            max_length=self.max_prompt_length,
            for_inference=False,
            use_diffusion=self.use_diffusion,
            bert_tokenizer=self.bert_tokenizer,
            cond_max_length=self.cond_max_length,
        )

        output_dir = os.path.join(self.output_dir, datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))
        
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                logits = logits[0]
            return logits.argmax(dim=-1)
        
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            mask = (labels != -100)
            if preds.shape[1] > labels.shape[1]:
                preds = preds[:, :labels.shape[1]]
                
            correct = (preds[mask] == labels[mask]).sum()
            total = mask.sum()
            
            return {
                "mean_token_accuracy": correct / total if total > 0 else 0.0
            }
        
        args = SFTConfig(
            output_dir=output_dir,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            logging_steps=10,
            save_strategy="epoch",
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            tf32=torch.cuda.is_available() and torch.cuda.is_tf32_supported(),
            max_length=self.max_prompt_length,
            packing=False,
            report_to="none",
            remove_unused_columns=False,
            completion_only_loss=True,
            dataloader_num_workers=4,
        )

        optimizer = AdamW([
            {"params": self.model.diffusion_proj.parameters(), "lr": 1e-4},
            {"params": self.model.diffusion_model.parameters(), "lr": 1e-4},
            {"params": self.model.log_vars, "lr": 1e-3}
        ], lr=1e-4)

        data_collator = QAGDataCollator(pad_token_id=self.tokenizer.pad_token_id, completion_only_loss=args.completion_only_loss)

        trainer = QAGTrainer(
            model=self.model, 
            args=args,
            train_dataset=dataset_train,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer, 
            data_collator=data_collator,
            optimizers=(optimizer, None),
        )

        trainer.train()

    def start_inference(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        dataset_test = tokenize_dataset(
            Dataset.from_list(self.dataset_test),
            self.tokenizer,
            model_type=QAGRunner.instruction_type,
            max_length=self.max_prompt_length,
            for_inference=True,
            use_diffusion=self.use_diffusion,
            bert_tokenizer=self.bert_tokenizer,
            cond_max_length=self.cond_max_length,
        )

        os.makedirs(self.output_dir, exist_ok=True)
        out_path = os.path.join(self.output_dir, f"infer_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jsonl")

        with torch.no_grad(), open(out_path, "w", encoding="utf-8") as fout:
            for sample in tqdm(dataset_test, desc="Inferencing"):
                tokenized = self.tokenizer.apply_chat_template(
                    sample["prompt"],
                    add_generation_prompt=True,
                    tokenize=True,
                    padding=False,
                    truncation=True,
                    max_length=self.max_prompt_length,
                    return_tensors="pt",
                    return_dict=True,
                ).to(self.device)

                if self.use_diffusion:
                    inputs = {
                        "input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"],
                        "x_input_ids": torch.tensor([sample["x_input_ids"]], device=self.device), 
                        "x_input_mask": torch.tensor([sample["x_input_mask"]], device=self.device),
                        "x_input_attention_mask": torch.tensor([sample["x_input_attention_mask"]], device=self.device),
                        "diffusion_steps": self.diffusion_steps
                    }
                else:
                    inputs = {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"]}

                outputs = self.model.generate(
                    inputs=inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.10,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

                output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                output_text = [text.strip() for text in output_text]

                rec = { "context": sample.get("context", ""), "label": sample.get("label", ""), "prediction": output_text[0] if output_text else ""}
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def run(self):
        self._load_tokenizer()
        self._load_dataset()
        self._load_model()
        self._prepare_lora()

        if not (self.use_training or self.use_inference):
            raise ValueError("need --use_training or --use_inference")

        if self.use_training:
            self.start_training()
        if self.use_inference:
            self.start_inference()

def main(dataset_name, train_data_path, test_data_path, **kwargs):
    runner = QAGRunner(dataset_name, {"train_data_path": train_data_path, "test_data_path": test_data_path}, **kwargs)
    runner.run()

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="DiFRa QAGRunner")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--train_data_path", required=True)
    p.add_argument("--test_data_path", required=True)
    p.add_argument("--max_prompt_length", type=int, default=4096)
    p.add_argument("--max_qa_pair", type=int, default=10)
    p.add_argument("--infer_max_qa_pair", type=int, default=10)
    p.add_argument("--model_type", choices=["instruct", "chat"], required=True)
    p.add_argument("--model_name", required=True)
    p.add_argument("--load_model_accuracy", default="bf16")
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--load_from_pretrained", action="store_true")
    p.add_argument("--pretrained_model_name")
    p.add_argument("--freeze_llm", action="store_true")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--use_lora_on_llama_model", action="store_true")
    p.add_argument("--use_lora_on_denoiser", action="store_true")
    p.add_argument("--use_training", action="store_true")
    p.add_argument("--use_inference", action="store_true")
    p.add_argument("--use_concepts", action="store_true")
    p.add_argument("--num_concepts", type=int)
    p.add_argument("--use_diffusion", action="store_true")
    p.add_argument("--diffusion_steps", type=int, default=25)
    p.add_argument("--bert_model_name")
    p.add_argument("--cond_max_length", type=int)
    p.add_argument("--use_ema", action="store_true")
    p.add_argument("--lambda_diff", type=float, default=0.1)
    p.add_argument("--diffusion_mlp_block_num", type=int, default=2)
    p.add_argument("--use_knowledge_graph", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()
    set_seed(args.seed)
    main(**vars(args))