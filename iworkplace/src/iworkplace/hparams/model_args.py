from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    
    model_type: str = field(default="instruct", metadata={"help": "The type of the model (e.g., instruct, chat)."})
    model_name: str = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."})
    load_model_accuracy: str = field(default="bf16", metadata={"help": "Model loading precision."})
    
    # Checkpoint
    load_from_pretrained: bool = field(default=False, metadata={"help": "Whether to load from a specific pretrained checkpoint."})
    pretrained_model_name: Optional[str] = field(default=None, metadata={"help": "Path to the specific pretrained checkpoint."})
    
    # Diffusion & Concept
    use_diffusion: bool = field(default=False, metadata={"help": "Whether to use diffusion modules."})
    diffusion_steps: int = field(default=25, metadata={"help": "Number of diffusion steps."})
    bert_model_name: Optional[str] = field(default=None, metadata={"help": "Name or path of the BERT model for diffusion conditioning."})
    use_ema: bool = field(default=False, metadata={"help": "Whether to use Exponential Moving Average for diffusion."})
    lambda_diff: float = field(default=0.1, metadata={"help": "Weight for the diffusion loss."})
    diffusion_mlp_block_num: int = field(default=2, metadata={"help": "Number of MLP blocks in the diffusion model."})