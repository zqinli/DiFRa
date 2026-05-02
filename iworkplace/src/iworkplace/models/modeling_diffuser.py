from dataclasses import dataclass
from typing import Optional, List
import functools
import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from .diffusion.transformer_model import TransformerNetModel
from .diffusion.gaussian_diffusion import SpacedDiffusion, space_timesteps
from .diffusion.step_sample import create_named_schedule_sampler, LossAwareSampler
from .diffusion import gaussian_diffusion as gd
from .diffusion.dpm_solver_pytorch import NoiseScheduleVP, model_wrapper, DPM_Solver
from .diffusion.utils.nn import update_ema

@dataclass
class DiffusionConfig:
    # Backbones
    bert_model_name: str = "bert-base-uncased"
    hidden_t_dim: int = 768
    hidden_dim: int = 768
    dropout: float = 0.1

    denoiser_use_fp32: bool = True
    dtype: str = "fp32"  # choices: "fp16", "bf16", "fp32"

    use_ema: bool = False
    ema_rate: float = 0.99

    init_pretrained: str = "bert"
    learned_mean_embed: bool = False

    # Diffusion 核心配置
    diffusion_steps: int = 1000
    noise_schedule: str = "cosine"
    timestep_respacing: str = ""
    rescale_timesteps: bool = True
    predict_xstart: bool = False
    learn_sigma: bool = False
    sigma_small: bool = False
    use_kl: bool = False
    rescale_learned_sigmas: bool = False
    schedule_sampler: str = "lossaware"
    rejection_rate: float = 0.0
    denoise: bool = False
    denoise_rate: float = 0.0
    device: str = ""
    _attn_implementation: str = "eager"

    dpm_solver_order: int = 3
    dpm_solver_steps: Optional[int] = 25
    dpm_solver_t_start: float = 1.0
    dpm_solver_t_end: float = 1e-4
    dpm_solver_skip_type: str = "time_uniform"
    dpm_solver_method: str = "multistep"
    dpm_solver_lower_order_final: bool = False
    dpm_solver_guidance_type: str = "uncond"
    dpm_solver_guidance_scale: float = 1.0

    is_spaced_diffusion: bool = False
    timestep_map: Optional[List[int]] = None


class ConceptDiffusion(nn.Module):

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)
        self.pad_id, self.cls_id, self.sep_id = (getattr(self.tokenizer, attr, default) for attr, default in [("pad_token_id", 0), ("cls_token_id", 101), ("sep_token_id", 102)])
    
        self.hidden = int(config.hidden_dim if not config.learn_sigma else config.hidden_dim * 2)

        self.denoiser = TransformerNetModel(
            input_dims=config.hidden_dim,
            output_dims=(config.hidden_dim if not config.learn_sigma else config.hidden_dim * 2),
            hidden_t_dim=config.hidden_t_dim,
            dropout=config.dropout,
            config_name=config.bert_model_name,
            init_pretrained=config.init_pretrained,
            vocab_size=self.tokenizer.vocab_size,
            learned_mean_embed=config.learned_mean_embed,
            _attn_implementation=config._attn_implementation
        )

        betas = gd.get_named_beta_schedule(config.noise_schedule, config.diffusion_steps, warmup_steps_ratio=0.1)

        if config.timestep_respacing == "":
            config.timestep_respacing = [config.diffusion_steps]

        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(config.diffusion_steps, config.timestep_respacing),
            betas=betas,
            rescale_timesteps=config.rescale_timesteps,
            predict_xstart=config.predict_xstart,
            learn_sigmas=config.learn_sigma,
            sigma_small=config.sigma_small,
            use_kl=config.use_kl,
            rescale_learned_sigmas=config.rescale_learned_sigmas,
            rejection_rate=config.rejection_rate,
            denoise=config.denoise,
            denoise_rate=config.denoise_rate,
            device=config.device,
            max_T=config.diffusion_steps,
        )

        self.config.is_spaced_diffusion = True
        self.config.timestep_map = self.diffusion.timestep_map

        self.schedule_sampler = create_named_schedule_sampler(config.schedule_sampler, self.diffusion)

        self.compute_dtype = (
            torch.float16 if (config.dtype == "fp16" and torch.cuda.is_available()) 
            else (torch.bfloat16 if (config.dtype == "bf16" and torch.cuda.is_available()) 
                else torch.float32)
        )

        if config.denoiser_use_fp32:
            self.denoiser = self.denoiser.to(dtype=torch.float32)
        else:
            self.denoiser = self.denoiser.to(dtype=self.compute_dtype)        

        print(f"denoiser use dtype: {next(self.denoiser.parameters()).dtype}")

        self.ema_rate = config.ema_rate
        self.denoiser_ema = None

        if self.config.use_ema and self.ema_rate > 0:
            self.denoiser_ema = TransformerNetModel(
                input_dims=config.hidden_dim,
                output_dims=(config.hidden_dim if not config.learn_sigma else config.hidden_dim * 2),
                hidden_t_dim=config.hidden_t_dim,
                dropout=config.dropout,
                config_name=config.bert_model_name,
                init_pretrained=config.init_pretrained,
                vocab_size=self.tokenizer.vocab_size,
                learned_mean_embed=config.learned_mean_embed,
                _attn_implementation=config._attn_implementation
            )

            if config.denoiser_use_fp32:
                self.denoiser_ema = self.denoiser_ema.to(dtype=torch.float32)
            else:
                self.denoiser_ema = self.denoiser_ema.to(dtype=self.compute_dtype)

            self.denoiser_ema.load_state_dict(self.denoiser.state_dict())
            self.denoiser_ema.eval()

            for param in self.denoiser_ema.parameters():
                param.requires_grad = False 

            print(f"denoiser_ema (shadow model) created. dtype: {next(self.denoiser_ema.parameters()).dtype}")
            print("[INFO] EMA parameters frozen.")
        else:
            print("[INFO] EMA is DISABLED for denoiser.")

        self.noise_schedule = None
        self.model_kwargs = {}
        self.model_fn = None
        self.dpm_solver = None
        self._dpm_solver_is_ema = False



    @torch.no_grad()
    def _get_generated_emb(self, pred_xstart, input_ids, input_mask):
        B, L, H = pred_xstart.shape
        device = pred_xstart.device

        special_ids = {self.pad_id, self.cls_id, self.sep_id}
        special_mask = torch.isin(input_ids, torch.tensor(list(special_ids), device=device))

        trg_mask = (
            (input_ids != self.pad_id) 
            & (input_mask == 1) 
            & (~special_mask)
        ).to(torch.bool)

        max_n = int(trg_mask.sum(dim=1).amax().item())
        if max_n == 0:
            out = pred_xstart.new_zeros(B, 0, H)
            out_mask = torch.zeros(B, 0, dtype=torch.bool, device=device)
            return out, out_mask
        
        out = pred_xstart.new_zeros(B, max_n, H) 

        seq_indices = torch.cumsum(trg_mask, dim=1) - 1 
        batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(B, L)

        out[batch_indices[trg_mask], seq_indices[trg_mask]] = pred_xstart[trg_mask]

        lengths = trg_mask.sum(dim=1)
        ar = torch.arange(max_n, device=device).unsqueeze(0).expand(B, max_n)
        out_mask = ar < lengths.unsqueeze(1)

        return out, out_mask


    def forward(
        self,
        input_ids: Tensor,
        input_mask: Tensor,
        attention_mask: Tensor,
        **kwargs,
    ):
        B, L = input_ids.shape
        device = input_ids.device

        t, weights = self.schedule_sampler.sample(B, device) 

        model_kwargs = {
            'input_ids': input_ids,
            'input_mask': input_mask,
            'input_attention_mask': attention_mask
        }
        
        compute_losses = functools.partial(
            self.diffusion.training_losses,
            self.denoiser,
            t,
            model_kwargs=model_kwargs
        )

        losses = compute_losses()

        if isinstance(self.schedule_sampler, LossAwareSampler):
            self.schedule_sampler.update_with_local_losses(
                t, losses["loss"].detach()
            )
        loss = (losses["loss"] * weights).mean()

        pred_xstart = losses["pred_xstart"]
        out, out_mask = self._get_generated_emb(pred_xstart, input_ids, input_mask)

        return loss, out, out_mask


    @torch.inference_mode()
    def sample(
        self,
        input_ids: Tensor,
        input_mask: Tensor,
        attention_mask: Tensor,
        steps: Optional[int] = None,
    ):
        use_ema = self.config.use_ema and self.ema_rate > 0 and self.denoiser_ema is not None
        model_to_use = self.denoiser_ema if use_ema else self.denoiser
        model_to_use.eval()
        
        model_device = next(model_to_use.parameters()).device
        solver_is_correct_mode = (self._dpm_solver_is_ema == use_ema)
        
        self.model_kwargs['input_attention_mask'] = attention_mask.to(model_device)

        if self.dpm_solver is None or not solver_is_correct_mode:
            print(f"--- sampling... (Using {'EMA' if use_ema else 'LIVE'} weights) ---")
            print(f"[DPM_Solver Init] Denoiser ({'EMA' if use_ema else 'LIVE'}) compute_dtype: {next(model_to_use.parameters()).dtype}")

            self.noise_schedule = NoiseScheduleVP(
                schedule="discrete", 
                betas=torch.from_numpy(self.diffusion.betas)
            )
            
            self.model_fn = model_wrapper(
                model=model_to_use,
                noise_schedule=self.noise_schedule,
                model_type="noise" if not self.config.predict_xstart else "x_start",
                model_kwargs=self.model_kwargs, 
                guidance_type=self.config.dpm_solver_guidance_type,
            )
            
            self.dpm_solver = DPM_Solver(
                self.model_fn, 
                self.noise_schedule, 
                algorithm_type="dpmsolver++"
            )
            self._dpm_solver_is_ema = use_ema

        if self.dpm_solver is not None:
             self.model_kwargs['input_attention_mask'] = attention_mask.to(model_device)

        device = input_ids.device
        input_ids_x = input_ids
        x_start = model_to_use.get_embeds(input_ids_x)
        input_ids_mask = input_mask.clone()
        input_ids_mask_ori = input_ids_mask
        noise = torch.randn_like(x_start)
        input_ids_mask = torch.broadcast_to(input_ids_mask.unsqueeze(dim=-1), x_start.shape).to(device)
        x_noised = torch.where(input_ids_mask == 0, x_start, noise)
        
        with torch.autocast(device_type="cuda", dtype=self.compute_dtype):
            x_sample = self.dpm_solver.sample(
                x_noised,
                steps=steps or self.config.dpm_solver_steps,
                t_start=self.config.dpm_solver_t_start,
                t_end=self.config.dpm_solver_t_end,
                order=self.config.dpm_solver_order,
                skip_type=self.config.dpm_solver_skip_type,
                method=self.config.dpm_solver_method,
                input_ids_mask=input_ids_mask,
                x_start=x_start,
                lower_order_final=self.config.dpm_solver_lower_order_final,
            )
        
        out, out_mask = self._get_generated_emb(x_sample, input_ids, input_ids_mask_ori)
        return out, out_mask
    
    def update_ema(self):
        """
        Update EMA weights during training (补充稳健性优化)
        """
        if not self.training or not self.config.use_ema or self.ema_rate <= 0 or self.denoiser_ema is None:
            return
        
        live_denoiser = self.denoiser
        ema_denoiser = self.denoiser_ema
        ema_rate = self.ema_rate
        
        target_dict = ema_denoiser.state_dict()
        # 定义需要忽略的参数（根据你的模型结构调整，可选）
        ignore_keys = ["layer_norm", ".bias", "norm"]  # 跳过LayerNorm和偏置项

        if hasattr(live_denoiser, "merge_adapter"):
            try:
                live_denoiser.merge_adapter()
                source_model = live_denoiser.get_base_model()
                source_dict = source_model.state_dict()
                
                for key, targ_param in target_dict.items():
                    # 1. 跳过需要忽略的参数
                    if any(ignore_key in key for ignore_key in ignore_keys):
                        if key in source_dict or key.replace(".weight", ".base_layer.weight") in source_dict:
                            src_key = key.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
                            src_param = source_dict.get(src_key, source_dict.get(key))
                            targ_param.data.copy_(src_param.data)
                        continue
                    
                    # 2. 跳过非浮点型/不可训练参数
                    if not targ_param.is_floating_point() or not targ_param.requires_grad:
                        continue

                    source_key = key.replace(".weight", ".base_layer.weight").replace(".bias", ".base_layer.bias")
                    src_param = None
                    if source_key in source_dict:
                        src_param = source_dict[source_key]
                    elif key in source_dict:
                        src_param = source_dict[key]
                    
                    if src_param is not None:
                        # 3. 设备和dtype一致性检查
                        if targ_param.device != src_param.device:
                            targ_param = targ_param.to(src_param.device)
                        if targ_param.dtype != src_param.dtype:
                            src_param = src_param.to(targ_param.dtype)
                        # 核心更新逻辑
                        targ_param.data.mul_(ema_rate).add_(src_param.data, alpha=1 - ema_rate)
                        # 4. 强制冻结EMA参数
                        if targ_param.requires_grad:
                            targ_param.requires_grad = False
                        
            finally:
                live_denoiser.unmerge_adapter()
        else:
            source_dict = live_denoiser.state_dict()
            for key, targ_param in target_dict.items():
                # 1. 跳过需要忽略的参数
                if any(ignore_key in key for ignore_key in ignore_keys):
                    if key in source_dict:
                        targ_param.data.copy_(source_dict[key].data)
                    continue
                
                # 2. 跳过非浮点型/不可训练参数
                if not targ_param.is_floating_point() or not targ_param.requires_grad:
                    continue 

                if key in source_dict:
                    src_param = source_dict[key]
                    # 3. 设备和dtype一致性检查
                    if targ_param.device != src_param.device:
                        targ_param = targ_param.to(src_param.device)
                    if targ_param.dtype != src_param.dtype:
                        src_param = src_param.to(targ_param.dtype)
                    # 核心更新逻辑
                    targ_param.data.mul_(ema_rate).add_(src_param.data, alpha=1 - ema_rate)
                    # 4. 强制冻结EMA参数
                    if targ_param.requires_grad:
                        targ_param.requires_grad = False