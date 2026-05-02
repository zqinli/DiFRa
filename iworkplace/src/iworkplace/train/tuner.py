# src/iworkplace/train/tuner.py
import os
import json
import torch
from datetime import datetime
from datasets import Dataset
from trl import SFTConfig
from tqdm import tqdm

from iworkplace.hparams import DataArguments, ModelArguments, FinetuningArguments
from iworkplace.utils.loader import load_tokenizer, load_model
from iworkplace.data.loader import get_dataset

from iworkplace.utils.dataset_utils import tokenize_dataset
from iworkplace.utils.trainer import QAGTrainer
from iworkplace.utils.data_collator import QAGDataCollator

def run_train(
    model_args: ModelArguments,
    data_args: DataArguments,
    finetuning_args: FinetuningArguments,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 1,
    output_dir: str = "./outputs",
    seed: int = 42,
    do_infer_after_train: bool = True,
    deepspeed: str = None,  # 新增 DeepSpeed 配置文件路径参数
):
    rank = os.environ.get("LOCAL_RANK", "0")
    is_main_process = (rank == "0")
    
    if is_main_process:
        print(f"启动训练任务 | 模型: {model_args.model_name} | 数据集: {data_args.dataset_name}")

    tokenizer, bert_tokenizer, unk_token_id = load_tokenizer(model_args)
    raw_dataset_train, raw_dataset_test = get_dataset(data_args, model_args, bert_tokenizer)
    
    if is_main_process:
        print(f"数据加载完成: 训练集 {len(raw_dataset_train)} 条，测试集 {len(raw_dataset_test)} 条")

    model = load_model(model_args, finetuning_args, tokenizer, unk_token_id, is_main_process)
    
    if is_main_process:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"可训练参数总量: {trainable_params:,} (~{trainable_params/1e6:.2f}M)")
        print("正在处理 Tokenize 数据集...")

    dataset_train = tokenize_dataset(
        Dataset.from_list(raw_dataset_train),
        tokenizer,
        model_type=model_args.model_type,
        max_length=data_args.max_prompt_length,
        for_inference=False,
        use_diffusion=model_args.use_diffusion,
        bert_tokenizer=bert_tokenizer,
        cond_max_length=data_args.cond_max_length,
    )

    dataset_test = tokenize_dataset(
        Dataset.from_list(raw_dataset_test),
        tokenizer,
        model_type=model_args.model_type,
        max_length=data_args.max_prompt_length,
        for_inference=False,
        use_diffusion=model_args.use_diffusion,
        bert_tokenizer=bert_tokenizer,
        cond_max_length=data_args.cond_max_length,
    )

    run_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    final_output_dir = os.path.join(output_dir, run_name)

    sft_args = SFTConfig(
        output_dir=final_output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=8,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        tf32=torch.cuda.is_available() and torch.cuda.is_tf32_supported(),
        max_length=data_args.max_prompt_length,
        packing=False,
        report_to="none",
        remove_unused_columns=False,
        completion_only_loss=True,
        dataloader_num_workers=4,
        seed=seed,
        deepspeed=deepspeed,  # 将配置文件路径传给 SFTConfig
        # activation_offloading=True,
    )

    data_collator = QAGDataCollator(
        pad_token_id=tokenizer.pad_token_id, 
        completion_only_loss=sft_args.completion_only_loss
    )

    # 移除了手动的 optimizers=(optimizer, None)
    trainer = QAGTrainer(
        model=model, 
        args=sft_args,
        train_dataset=dataset_train,
        eval_dataset=dataset_test,
        processing_class=tokenizer, 
        data_collator=data_collator,
    )

    if is_main_process:
        print("配置完毕，正式启动训练循环...")
        
    trainer.train()

    trainer.save_model(final_output_dir)

    if do_infer_after_train and is_main_process:
        print("\n训练结束，准备使用当前显存中的权重进行直接推理...")
        
        infer_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
        infer_model.eval()
        device = next(infer_model.parameters()).device
        
        if getattr(infer_model.config, "use_ema", False) and hasattr(infer_model, "diffusion_model"):
            if hasattr(infer_model.diffusion_model, "swap_to_ema_weights"):
                infer_model.diffusion_model.swap_to_ema_weights()

        infer_dataset = tokenize_dataset(
            Dataset.from_list(raw_dataset_test),
            tokenizer,
            model_type=model_args.model_type,
            max_length=data_args.max_prompt_length,
            for_inference=True,
            use_diffusion=model_args.use_diffusion,
            bert_tokenizer=bert_tokenizer,
            cond_max_length=data_args.cond_max_length,
        )

        os.makedirs(final_output_dir, exist_ok=True)
        out_path = os.path.join(final_output_dir, f"infer_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.jsonl")

        with torch.no_grad(), open(out_path, "w", encoding="utf-8") as fout:
            for sample in tqdm(infer_dataset, desc="Inferencing"):
                tokenized = tokenizer.apply_chat_template(
                    sample["prompt"],
                    add_generation_prompt=True,
                    tokenize=True,
                    padding=False,
                    truncation=True,
                    max_length=data_args.max_prompt_length,
                    return_tensors="pt",
                    return_dict=True,
                ).to(device)

                if model_args.use_diffusion:
                    inputs = {
                        "input_ids": tokenized["input_ids"], 
                        "attention_mask": tokenized["attention_mask"],
                        "x_input_ids": torch.tensor([sample["x_input_ids"]], device=device), 
                        "x_input_mask": torch.tensor([sample["x_input_mask"]], device=device),
                        "x_input_attention_mask": torch.tensor([sample["x_input_attention_mask"]], device=device),
                        "diffusion_steps": model_args.diffusion_steps
                    }
                else:
                    inputs = {
                        "input_ids": tokenized["input_ids"], 
                        "attention_mask": tokenized["attention_mask"]
                    }

                outputs = infer_model.generate(
                    inputs=inputs,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.95,
                    repetition_penalty=1.10,
                    pad_token_id=tokenizer.eos_token_id,
                )

                output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                output_text = [text.strip() for text in output_text]

                rec = {
                    "context": sample.get("context", ""), 
                    "label": sample.get("label", ""), 
                    "prediction": output_text[0] if output_text else ""
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")