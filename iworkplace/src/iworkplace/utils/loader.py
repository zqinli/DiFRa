# src/iworkplace/model/loader.py
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from iworkplace.hparams import ModelArguments, FinetuningArguments

# 假设旧代码的 models 目录已迁移至 src/iworkplace/models
from iworkplace.models.modeling_qag import QAGConfig, QAGForCausalLM

UNK_TOKEN = "<unk>"

def load_tokenizer(model_args: ModelArguments):
    tokenizer_src = (
        model_args.pretrained_model_name 
        if model_args.load_from_pretrained and model_args.pretrained_model_name 
        else model_args.model_name
    )
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    unk_token_id = 0
    if model_args.use_diffusion:
        tokenizer.add_special_tokens({"additional_special_tokens": [UNK_TOKEN]})
        unk_token_id = tokenizer.convert_tokens_to_ids(UNK_TOKEN)
    
    bert_tokenizer = None
    if model_args.bert_model_name:
        bert_tokenizer = AutoTokenizer.from_pretrained(model_args.bert_model_name)
        
    return tokenizer, bert_tokenizer, unk_token_id


def load_model(
    model_args: ModelArguments, 
    finetuning_args: FinetuningArguments, 
    tokenizer, 
    unk_token_id: int,
    is_main_process: bool = True
):
    if model_args.load_from_pretrained:
        if is_main_process:
            print(f"加载预训练模型权重: {model_args.pretrained_model_name}")
        cfg = QAGConfig.from_pretrained(model_args.pretrained_model_name)
        model = QAGForCausalLM.from_pretrained(model_args.pretrained_model_name, config=cfg, trust_remote_code=True)
    else:
        cfg = QAGConfig(
            model_name_or_path=model_args.model_name,
            load_model_accuracy=model_args.load_model_accuracy,
            freeze_llm=finetuning_args.freeze_llm,
            use_diffusion=model_args.use_diffusion,
            bert_model_name_or_path=model_args.bert_model_name,
            num_concepts=None,  # 根据需要可以从 data_args 传入
            unk_token=UNK_TOKEN,
            unk_token_id=unk_token_id,
            use_flash_att=False,
            use_ema=model_args.use_ema,
            lambda_diff=model_args.lambda_diff,
            diffusion_mlp_block_num=model_args.diffusion_mlp_block_num,
        )
        model = QAGForCausalLM(cfg)

    if tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        model.config.pad_token_id = tokenizer.eos_token_id

    vocab_size = len(tokenizer)
    model.llama_model.resize_token_embeddings(vocab_size)
    model.llama_model.config.vocab_size = vocab_size

    if finetuning_args.use_lora:
        if is_main_process:
            print("正在注入 LoRA 适配器...")
            
        if finetuning_args.use_lora_on_llama_model:
            lora_cfg = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                inference_mode=False,
                target_modules=["q_proj", "v_proj"],
                task_type=TaskType.CAUSAL_LM,
            )
            model.llama_model = get_peft_model(model.llama_model, lora_cfg)
            if is_main_process:
                model.llama_model.print_trainable_parameters()
                
        if finetuning_args.use_lora_on_denoiser and hasattr(model, "diffusion_model"):
            lora_cfg_denoiser = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.1,
                bias="none",
                inference_mode=False,
                target_modules=["query", "value"],
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            model.diffusion_model.denoiser = get_peft_model(
                model.diffusion_model.denoiser, 
                lora_cfg_denoiser
            )
            
            for name, param in model.diffusion_model.denoiser.named_parameters():
                if "time_embed" in name or "time_modulator" in name:
                    param.requires_grad = True

    return model