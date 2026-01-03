import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from peft import PeftModel
from pathlib import Path
import os
import json
from dataclasses import asdict
import time
import numpy as np
import torch.nn.functional as F

from .modeling_diffuser import ConceptDiffusion, DiffusionConfig


class QAGConfig(PretrainedConfig):
    model_type = "qag"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-3.2-1B-Instruct",
        load_model_accuracy: str = "fp16",
        freeze_llm: bool = False,
        use_diffusion: bool = False,
        bert_model_name_or_path: str | None = None,
        num_concepts: int = 16,
        unk_token: str = "<unk>",
        unk_token_id: int = 0,
        use_flash_att: bool = False,
        use_ema: bool = False,
        lambda_diff: float = 0.1,
        diffusion_mlp_block_num: int = 2,
        **kwargs,
    ):  
        self.model_name_or_path = model_name_or_path
        self.load_model_accuracy = load_model_accuracy
        self.freeze_llm = freeze_llm
        self.use_diffusion = use_diffusion
        self.bert_model_name_or_path = bert_model_name_or_path
        self.num_concepts = num_concepts
        self.unk_token = unk_token
        self.unk_token_id = unk_token_id
        self.use_flash_att = use_flash_att
        self.use_ema = use_ema
        self.lambda_diff = lambda_diff
        self.diffusion_mlp_block_num = diffusion_mlp_block_num
        super().__init__(**kwargs)


class QAGForCausalLM(PreTrainedModel):
    config_class = QAGConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: QAGConfig):
        super().__init__(config)

        acc = config.load_model_accuracy.lower()
        if acc not in {"fp16", "bf16", "int8", "int4"}:
            raise ValueError("load_model_accuracy must be one of fp16/bf16/int8/int4")

        quant_cfg = None
        dtype = torch.float16 if acc == "fp16" else torch.bfloat16
        if acc in {"int8", "int4"}:
            quant_cfg = BitsAndBytesConfig(
                load_in_8bit=acc == "int8",
                load_in_4bit=acc == "int4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        
        model_kwargs = {
            "device_map": "auto",
            "dtype": dtype,
            "quantization_config": quant_cfg,
        }
        
        if config.use_flash_att:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            config.model_name_or_path, 
            **model_kwargs
        )
        
        if config.freeze_llm:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False

        self.use_diffusion = config.use_diffusion
        if self.use_diffusion:
            if getattr(config, "diffusion_config", None) and os.path.exists(config.diffusion_config):
                with open(config.diffusion_config, "r") as f:
                    diff_cfg = json.load(f)
                diff_cfg["device"] = str(self.llama_model.device)
                diff_cfg.pop("denoiser_is_lora", None)
                self.diffusion_config = DiffusionConfig(**diff_cfg)
            else:
                self.diffusion_config = DiffusionConfig(
                    bert_model_name=config.bert_model_name_or_path,
                    device=self.llama_model.device,
                    use_ema=config.use_ema,
                )
            self.diffusion_model = ConceptDiffusion(self.diffusion_config).to(device=self.llama_model.device)

            mlp_block_num = config.diffusion_mlp_block_num
            if mlp_block_num < 0:
                raise ValueError("diffusion_mlp_block_num must be >= 0")
            
            diffusion_proj_layers = []
            diffusion_proj_layers.append(nn.Linear(self.diffusion_model.hidden, self.llama_model.config.hidden_size))
            for _ in range(mlp_block_num):
                diffusion_proj_layers.append(nn.GELU())
                diffusion_proj_layers.append(nn.Dropout(p=0.1))
                diffusion_proj_layers.append(nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size)) 
            diffusion_proj_layers.append(nn.LayerNorm(self.llama_model.config.hidden_size))   
            
            self.diffusion_proj = nn.Sequential(*diffusion_proj_layers).to(dtype=self.llama_model.dtype).to(device=self.llama_model.device)
            self.log_vars = nn.Parameter(torch.zeros(2))

    def encode_inputs(
        self,
        input_ids: torch.Tensor,
        *,
        x_input_ids=None,
        x_input_mask=None,
        x_input_attention_mask=None,
        inference: bool = False,
        diffusion_steps: int = 15
    ):
        diffusion_loss = None
        input_embeds = self.llama_model.get_input_embeddings()(input_ids)

        B, T, H = input_embeds.shape
        if H != self.llama_model.config.hidden_size:
            raise ValueError(f"{H}")
        if not self.use_diffusion:
            return input_embeds, diffusion_loss
        
        if inference:
            with torch.autocast(device_type="cuda"):
                y0_pred, y0_mask = self.diffusion_model.sample(
                    input_ids=x_input_ids, input_mask=x_input_mask, attention_mask=x_input_attention_mask, steps=diffusion_steps
                )
            diffusion_loss = None
        else:
            with torch.autocast(device_type="cuda"):
                diffusion_loss, y0_pred, y0_mask = self.diffusion_model(
                    input_ids=x_input_ids, input_mask=x_input_mask, attention_mask=x_input_attention_mask
                )

        inject_embeds = self.diffusion_proj(y0_pred.to(input_embeds.dtype))

        unk_token_id = self.config.unk_token_id
        updated_input_embeds = input_embeds.clone()
        replace_idx = torch.nonzero(input_ids==unk_token_id).squeeze()
        inject_embeds = inject_embeds[y0_mask.bool()]
        updated_input_embeds[replace_idx[:, 0], replace_idx[:, 1]] = inject_embeds.to(input_embeds.dtype)
        
        return updated_input_embeds.contiguous(), diffusion_loss

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        x_input_ids=None,
        x_input_mask=None,
        x_input_attention_mask=None,
        **kwargs,
    ):
        if 'inputs_embeds' in kwargs:
            _ = kwargs.pop('inputs_embeds')

        inputs_embeds, diffusion_loss = self.encode_inputs(
            input_ids=input_ids,
            x_input_ids=x_input_ids,
            x_input_mask=x_input_mask,
            x_input_attention_mask=x_input_attention_mask,
        )
        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        loss_main = outputs.loss                             
        loss_diff = diffusion_loss if diffusion_loss is not None else 0.0 

        precision = torch.exp(-self.log_vars)                
        weighted_main = 0.5 * precision[0] * loss_main + 0.5 * self.log_vars[0]
        weighted_diff = 0.5 * precision[1] * loss_diff + 0.5 * self.log_vars[1]
        total_loss = weighted_main + weighted_diff

        outputs.loss = total_loss
        
        outputs.extra_losses = {
            "loss_main": loss_main.detach(),
            "loss_diff": loss_diff.detach(),
            "sigma_main": torch.exp(0.5 * self.log_vars[0]).detach(),
            "sigma_diff": torch.exp(0.5 * self.log_vars[1]).detach(),
        }
        return outputs
    
    def generate(self, inputs: torch.Tensor, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        
        inputs_embeds, _ = self.encode_inputs(
            input_ids=input_ids,
            x_input_ids=inputs.get("x_input_ids", None),
            x_input_mask=inputs.get("x_input_mask", None),
            x_input_attention_mask=inputs.get("x_input_attention_mask", None),
            inference=True,
            diffusion_steps=inputs.get("diffusion_steps", 15),
        )

        if 'inputs_embeds' in kwargs:
            _ = kwargs.pop('inputs_embeds')
        
        return self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )

    def get_input_embeddings(self):
        return self.llama_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llama_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.llama_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.llama_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.llama_model.set_decoder(decoder)

    def get_decoder(self):
        return self.llama_model.get_decoder()

    def save_pretrained(self, save_directory: str, **kwargs):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(self.llama_model, PeftModel):
            self.llama_model.save_pretrained(save_dir)
            super().save_pretrained(save_dir, **kwargs)
        elif not getattr(self.config, "freeze_llm", False):
            self.config.save_pretrained(save_directory)
            model_to_save = self.llama_model.module if hasattr(self.llama_model, 'module') else self.llama_model
            model_to_save.save_pretrained(save_directory)
        else:
            self.config.save_pretrained(save_directory)

        if getattr(self, "use_diffusion", False) and getattr(self, "diffusion_model", None) is not None:
            denoiser_is_lora = isinstance(self.diffusion_model.denoiser, PeftModel)
            
            model_dict = {"diffusion_proj": self.diffusion_proj.state_dict(),}

            if self.config.use_ema and hasattr(self.diffusion_model, "denoiser_ema"):
                model_dict["denoiser_ema"] = self.diffusion_model.denoiser_ema.state_dict()
            
            if denoiser_is_lora:
                denoiser_adapter_dir = save_dir / "diffusion_denoiser_adapters"
                self.diffusion_model.denoiser.save_pretrained(denoiser_adapter_dir)
            else:
                model_dict["diffusion_model"] = self.diffusion_model.state_dict()
                
            torch.save(model_dict, os.path.join(save_directory, "QAGLlama_pytorch_model.bin"))
            
            diffusion_config = asdict(self.diffusion_model.config)
            diffusion_config["denoiser_is_lora"] = denoiser_is_lora
            with open(os.path.join(save_directory, "diffusion_config.json"), "w", encoding="utf-8") as f:
                json.dump(diffusion_config, f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_path, *model_args, **kwargs)
        config = model.config

        denoiser_is_lora = False
        diffusion_json = os.path.join(pretrained_model_path, "diffusion_config.json")

        if os.path.exists(diffusion_json):
            config.diffusion_config = diffusion_json
            with open(diffusion_json, "r") as f:
                diff_cfg_json = json.load(f)
                denoiser_is_lora = diff_cfg_json.get("denoiser_is_lora", False)

        if getattr(config, "use_diffusion", False):
            model_bin_path = os.path.join(pretrained_model_path, "QAGLlama_pytorch_model.bin")
            
            if not os.path.exists(model_bin_path):
                 raise IOError(f"Diffusion model weights not found at {model_bin_path}")
            other_model_dict = torch.load(model_bin_path, map_location="cpu")
            
            if denoiser_is_lora:
                denoiser_adapter_dir = os.path.join(pretrained_model_path, "diffusion_denoiser_adapters")
                if not os.path.exists(denoiser_adapter_dir):
                    raise IOError(f"LoRA flag is True, but adapter directory not found: {denoiser_adapter_dir}")
                
                model.diffusion_model.denoiser = PeftModel.from_pretrained(
                    model.diffusion_model.denoiser,
                    denoiser_adapter_dir,
                    is_trainable=False
                )
                model.diffusion_proj.load_state_dict(other_model_dict["diffusion_proj"])
            else:
                model.diffusion_model.load_state_dict(other_model_dict["diffusion_model"])
                model.diffusion_proj.load_state_dict(other_model_dict["diffusion_proj"])

            if config.use_ema and "denoiser_ema" in other_model_dict:
                if hasattr(model.diffusion_model, "denoiser_ema"):
                    model.diffusion_model.denoiser_ema.load_state_dict(other_model_dict["denoiser_ema"])

        return model