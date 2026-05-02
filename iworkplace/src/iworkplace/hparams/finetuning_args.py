from dataclasses import dataclass, field

@dataclass
class FinetuningArguments:
    """Arguments pertaining to parameter-efficient fine-tuning (PEFT)."""
    
    freeze_llm: bool = field(default=False, metadata={"help": "Whether to completely freeze the LLM backbone."})
    use_lora: bool = field(default=False, metadata={"help": "Whether to use LoRA (Low-Rank Adaptation)."})
    use_lora_on_llama_model: bool = field(default=False, metadata={"help": "Whether to apply LoRA to the LLaMA base model."})
    use_lora_on_denoiser: bool = field(default=False, metadata={"help": "Whether to apply LoRA to the diffusion denoiser."})