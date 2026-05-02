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
import os, json
from dataclasses import asdict

from .modeling_diffuser import ConceptDiffusion, DiffusionConfig
import time
import numpy as np
import torch.nn.functional as F

def compute_kl_loss(p, q, pad_mask=None):
    """计算双向 KL 散度"""
    # p, q 形状: [Batch, Seq, Vocab]
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    # 只需要计算非 Padding (labels != -100) 的部分
    if pad_mask is not None:
        p_loss.masked_fill_(~pad_mask.unsqueeze(-1), 0.0)
        q_loss.masked_fill_(~pad_mask.unsqueeze(-1), 0.0)
        
    # 求平均：sum / 有效token数 / 词表大小 (kl_div默认会对最后一位求和)
    return (p_loss.sum() + q_loss.sum()) / 2.0 / pad_mask.sum()

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
            raise ValueError("load_model_accuracy 仅支持 fp16/bf16/int8/int4")

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
            # "device_map": "auto",
            # "attn_implementation": "eager",  # 默认实现
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
                print(f"[INFO] Loading diffusion config from {config.diffusion_config}")
                with open(config.diffusion_config, "r") as f:
                    diff_cfg = json.load(f)
                diff_cfg["device"] = str(self.llama_model.device)
                diff_cfg.pop("denoiser_is_lora", None)
                self.diffusion_config = DiffusionConfig(**diff_cfg)
            else:
                print("[INFO] Building new diffusion config")
                self.diffusion_config = DiffusionConfig(
                    bert_model_name=config.bert_model_name_or_path,
                    device=self.llama_model.device,
                    use_ema=config.use_ema,
                )
            self.diffusion_model = ConceptDiffusion(self.diffusion_config).to(device=self.llama_model.device)

            mlp_block_num = config.diffusion_mlp_block_num
            if mlp_block_num < 0:
                raise ValueError("diffusion_mlp_block_num 必须大于等于0")
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


    # def forward(
    #     self,
    #     input_ids=None,
    #     attention_mask=None,
    #     labels=None,
    #     x_input_ids=None,
    #     x_input_mask=None,
    #     x_input_attention_mask=None,
    #     **kwargs,
    # ):
    #     if 'inputs_embeds' in kwargs:
    #         _ = kwargs.pop('inputs_embeds')

    #     inputs_embeds, diffusion_loss = self.encode_inputs(
    #         input_ids=input_ids,
    #         x_input_ids=x_input_ids,
    #         x_input_mask=x_input_mask,
    #         x_input_attention_mask=x_input_attention_mask,
    #     )

    #     outputs = self.llama_model(
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask,
    #         labels=labels,
    #         **kwargs,
    #     )

    #     lambda_diff = self.config.lambda_diff
    #     total_loss = (1 - lambda_diff) * outputs.loss + (lambda_diff * diffusion_loss if diffusion_loss is not None else 0.0)
    #     outputs.loss = total_loss
    #     return outputs
    
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

        # ===== 原有：编码 + 取两路损失 =====
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
        loss_main = outputs.loss                             # L₁
        loss_diff = diffusion_loss if diffusion_loss is not None else 0.0  # L₂

        # ===== 新：不确定性加权 =====
        precision = torch.exp(-self.log_vars)                # e^{-s_i}
        weighted_main = 0.5 * precision[0] * loss_main + 0.5 * self.log_vars[0]
        weighted_diff = 0.5 * precision[1] * loss_diff + 0.5 * self.log_vars[1]
        total_loss = weighted_main + weighted_diff

        outputs.loss = total_loss
        # 方便监控：把原始损失附带出去
        outputs.extra_losses = {
            "loss_main": loss_main.detach(),
            "loss_diff": loss_diff.detach(),
            "sigma_main": torch.exp(0.5 * self.log_vars[0]).detach(),
            "sigma_diff": torch.exp(0.5 * self.log_vars[1]).detach(),
        }
        return outputs
    
    # def forward(
    #     self,
    #     input_ids=None,
    #     attention_mask=None,
    #     labels=None,
    #     x_input_ids=None,
    #     x_input_mask=None,
    #     x_input_attention_mask=None,
    #     **kwargs,
    # ):
    #     # 兼容性处理
    #     if 'inputs_embeds' in kwargs:
    #         _ = kwargs.pop('inputs_embeds')

    #     # 1. 编码阶段 (Diffusion 部分)
    #     # 无论 R-Drop 跑几次前向，Diffusion 的特征注入只需做一次，
    #     # 因为输入的 Prompt/Concept 向量是确定的。
    #     inputs_embeds, diffusion_loss = self.encode_inputs(
    #         input_ids=input_ids,
    #         x_input_ids=x_input_ids,
    #         x_input_mask=x_input_mask,
    #         x_input_attention_mask=x_input_attention_mask,
    #     )

    #     # 2. Llama 第一次前向传播 (Path 1)
    #     outputs1 = self.llama_model(
    #         inputs_embeds=inputs_embeds,
    #         attention_mask=attention_mask,
    #         labels=labels,
    #         **kwargs,
    #     )
        
    #     loss_rdrop = torch.tensor(0.0, device=input_ids.device)
    #     loss_main = outputs1.loss

    #     # 3. 如果在训练模式且有 labels，执行 R-Drop 的第二次前向 (Path 2)
    #     if self.training and labels is not None:
    #         outputs2 = self.llama_model(
    #             inputs_embeds=inputs_embeds,
    #             attention_mask=attention_mask,
    #             labels=labels,
    #             **kwargs,
    #         )
            
    #         # 平均两路主损失 (L1)
    #         loss_main = (outputs1.loss + outputs2.loss) / 2.0
            
    #         # 计算一致性损失 (L3)
    #         pad_mask = (labels != -100)
    #         loss_rdrop = compute_kl_loss(outputs1.logits, outputs2.logits, pad_mask)

    #     # 知识注入损失 (L2)
    #     loss_diff = diffusion_loss if diffusion_loss is not None else torch.tensor(0.0, device=input_ids.device)

    #     # 4. 三任务自适应加权逻辑
    #     # log_vars[0]: Main, log_vars[1]: Diff, log_vars[2]: R-Drop
    #     precision = torch.exp(-self.log_vars)
        
    #     # 任务 1: QA 生成精度
    #     weighted_main = 0.5 * precision[0] * loss_main + 0.5 * self.log_vars[0]
        
    #     # 任务 2: 知识对齐精度
    #     weighted_diff = 0.5 * precision[1] * loss_diff + 0.5 * self.log_vars[1]
        
    #     # 任务 3: 逻辑一致性精度 (R-Drop)
    #     if self.training and labels is not None:
    #         weighted_rdrop = 0.5 * precision[2] * loss_rdrop + 0.5 * self.log_vars[2]
    #     else:
    #         weighted_rdrop = 0.0

    #     total_loss = weighted_main + weighted_diff + weighted_rdrop

    #     # 5. 封装最终输出
    #     # 我们基于 outputs1 进行封装返回
    #     outputs = outputs1
    #     outputs.loss = total_loss
        
    #     # 附加 extra_losses 用于日志监控 (重点关注 sigma 是否收敛)
    #     # sigma = exp(0.5 * log_var) 代表模型自动学习到的每个任务的噪声标准差
    #     outputs.extra_losses = {
    #         "loss_main": loss_main.detach(),
    #         "loss_diff": loss_diff.detach(),
    #         "loss_rdrop": loss_rdrop.detach() if isinstance(loss_rdrop, torch.Tensor) else 0.0,
    #         "sigma_main": torch.exp(0.5 * self.log_vars[0]).detach(),
    #         "sigma_diff": torch.exp(0.5 * self.log_vars[1]).detach(),
    #         "sigma_rdrop": torch.exp(0.5 * self.log_vars[2]).detach(),
    #     }

    #     return outputs
    
    def generate(self, inputs: torch.Tensor, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        # print(len(input_ids[0]))
        inputs_embeds, _ = self.encode_inputs(
            input_ids=input_ids,
            x_input_ids=inputs.get("x_input_ids", None),
            x_input_mask=inputs.get("x_input_mask", None),
            x_input_attention_mask=inputs.get("x_input_attention_mask", None),
            inference=True,
            diffusion_steps=inputs.get("diffusion_steps", 15),
        )
        # print(f"DEBUG: inputs_embeds 的形状: {inputs_embeds.shape}")

        if 'inputs_embeds' in kwargs:
            _ = kwargs.pop('inputs_embeds')
        
        return self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            # output_attentions=True,
            # return_dict_in_generate=True,
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
                print(f"[INFO] Saved EMA denoiser weights")
            
            if denoiser_is_lora:
                denoiser_adapter_dir = save_dir / "diffusion_denoiser_adapters"
                print(f"Saving denoiser LoRA adapters to {denoiser_adapter_dir}")
                self.diffusion_model.denoiser.save_pretrained(denoiser_adapter_dir)
            
            else:
                print("Saving full diffusion model state dict (live weights)...")
                model_dict["diffusion_model"] = self.diffusion_model.state_dict()
                

            torch.save(model_dict, os.path.join(save_directory, "QAGLlama_pytorch_model.bin"))
            
            diffusion_config = asdict(self.diffusion_model.config)
            diffusion_config["denoiser_is_lora"] = denoiser_is_lora
            with open(os.path.join(save_directory, "diffusion_config.json"), "w", encoding="utf-8") as f:
                json.dump(diffusion_config, f, indent=2, ensure_ascii=False, default=str)

    @classmethod
    def from_pretrained(cls, pretrained_model_path, *model_args, **kwargs):
        
        print(f"[INFO] Loading base model and Llama weights from {pretrained_model_path}...")
        

        model = super().from_pretrained(pretrained_model_path, *model_args, **kwargs)
        config = model.config


        denoiser_is_lora = False
        diffusion_json = os.path.join(pretrained_model_path, "diffusion_config.json")

        if os.path.exists(diffusion_json):
            config.diffusion_config = diffusion_json
            with open(diffusion_json, "r") as f:
                diff_cfg_json = json.load(f)
                denoiser_is_lora = diff_cfg_json.get("denoiser_is_lora", False)
                print(f"[INFO] Denoiser is configured with LoRA: {denoiser_is_lora}")

        if getattr(config, "use_diffusion", False):
            model_bin_path = os.path.join(pretrained_model_path, "QAGLlama_pytorch_model.bin")
            
            if not os.path.exists(model_bin_path):
                 raise IOError(f"Diffusion model weights not found at {model_bin_path}")
            other_model_dict = torch.load(model_bin_path, map_location="cpu")
            
            if denoiser_is_lora:
                denoiser_adapter_dir = os.path.join(pretrained_model_path, "diffusion_denoiser_adapters")
                if not os.path.exists(denoiser_adapter_dir):
                    raise IOError(f"LoRA flag is True, but adapter directory not found: {denoiser_adapter_dir}")
                
                print(f"Loading denoiser LoRA adapters from {denoiser_adapter_dir}...")
                model.diffusion_model.denoiser = PeftModel.from_pretrained(
                    model.diffusion_model.denoiser,
                    denoiser_adapter_dir,
                    is_trainable=False
                )
                
                model.diffusion_proj.load_state_dict(other_model_dict["diffusion_proj"])
                
            else:
                print("Loading full diffusion model state dict (live weights)...")
                model.diffusion_model.load_state_dict(other_model_dict["diffusion_model"])
                model.diffusion_proj.load_state_dict(other_model_dict["diffusion_proj"])

            if config.use_ema and "denoiser_ema" in other_model_dict:
                if hasattr(model.diffusion_model, "denoiser_ema"):
                    model.diffusion_model.denoiser_ema.load_state_dict(other_model_dict["denoiser_ema"])
                    print(f"[INFO] Loaded EMA denoiser weights")
                else:
                    print(f"[WARNING] EMA weights found but denoiser_ema not initialized")


        return model
    