# src/iworkplace/utils/trainer.py
from typing import Union
from trl import SFTTrainer
from datasets import Dataset, DatasetDict, concatenate_datasets
import torch
from transformers import Trainer

class QAGTrainer(SFTTrainer):
    
    def _prepare_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        processing_class,
        training_args,
        packing: bool = False,
        formatting_function=None,
        dataset_name: str | None = None,
    ):
        column_names = dataset.column_names
        if "messages" in column_names:
            columns_to_process = ["messages"]
        elif {"prompt", "completion"}.issubset(column_names):
            columns_to_process = ["prompt", "completion"]
        else:
            raise ValueError(f"Unrecognized columns in dataset: {column_names}")

        columns_to_keep = [column for column in column_names if column not in columns_to_process]
        dataset_to_keep = dataset.remove_columns(columns_to_process) if columns_to_keep else None

        processed_dataset = super()._prepare_dataset(
            dataset.select_columns(columns_to_process),
            processing_class,
            training_args,
            packing=packing,
            formatting_func=formatting_function,
            dataset_name=dataset_name,
        )

        if dataset_to_keep is not None:
            if dataset_to_keep.num_rows != processed_dataset.num_rows:
                raise ValueError(
                    "Row mismatch after processing; packing or formatting_function may have altered row count."
                )
            processed_dataset = concatenate_datasets([processed_dataset, dataset_to_keep], axis=1)
            
        return processed_dataset
    

    def training_step(self, *args, **kwargs):
        with self.maybe_activation_offload_context:
            loss = super().training_step(*args, **kwargs)

        if self.model.training and hasattr(self.model, "diffusion_model"):
            self.model.diffusion_model.update_ema()

        return loss
    
    def log(self, logs: dict[str, float], *args, **kwargs) -> None:
        """
        增加 *args 和 **kwargs 以匹配 transformers.Trainer.log 的签名
        """
        if hasattr(self.model, "log_vars"):
            # 使用 torch.no_grad() 确保不会干扰梯度计算
            with torch.no_grad():
                log_vars = self.model.log_vars.detach().cpu()
                
                # 计算实际权重系数: 0.5 * exp(-log_var)
                weight_main = (0.5 * torch.exp(-log_vars[0])).item()
                weight_diff = (0.5 * torch.exp(-log_vars[1])).item()

                # 放入 logs 字典中，Trainer 会自动处理这些指标
                logs["weight/main"] = weight_main
                logs["weight/diff"] = weight_diff
                
                # 可选：记录 sigma 值
                logs["sigma/main"] = torch.exp(0.5 * log_vars[0]).item()
                logs["sigma/diff"] = torch.exp(0.5 * log_vars[1]).item()

        # 调用父类方法，注意透传所有参数
        super().log(logs, *args, **kwargs)

    def create_optimizer(self):
        """
        重写优化器创建逻辑，以便支持不同组件使用不同的学习率，
        同时让 HF Trainer 自动处理 DeepSpeed 的包装。
        """
        if self.optimizer is None:
            optimizer_grouped_parameters = []
            
            # 记录已经被分配了特殊学习率的参数 id，防止重复添加
            special_params_ids = set()
            
            # 1. 为 Diffusion 投影层分配学习率
            if hasattr(self.model, "diffusion_proj"):
                diffusion_proj_params = list(self.model.diffusion_proj.parameters())
                optimizer_grouped_parameters.append({"params": diffusion_proj_params, "lr": 1e-4})
                special_params_ids.update(id(p) for p in diffusion_proj_params)
                
            # 2. 为 Diffusion 模型分配学习率
            if hasattr(self.model, "diffusion_model"):
                diffusion_model_params = list(self.model.diffusion_model.parameters())
                optimizer_grouped_parameters.append({"params": diffusion_model_params, "lr": 1e-4})
                special_params_ids.update(id(p) for p in diffusion_model_params)
                
            # 3. 为 不确定性权重 (log_vars) 分配学习率
            # 注意：log_vars 是 nn.Parameter，需要放进列表中
            if hasattr(self.model, "log_vars"):
                optimizer_grouped_parameters.append({"params": [self.model.log_vars], "lr": 1e-3})
                special_params_ids.add(id(self.model.log_vars))
                
            # 4. 把剩下所有需要训练的参数（如 LLaMA 的 LoRA 参数）加入默认组
            main_params = [
                p for p in self.model.parameters() 
                if p.requires_grad and id(p) not in special_params_ids
            ]
            
            if main_params:
                # 使用 SFTConfig 里设置的主学习率 (默认就是你设定的 base learning rate)
                optimizer_grouped_parameters.append(
                    {"params": main_params, "lr": self.args.learning_rate}
                )

            # 5. 调用父类的辅助方法，获取环境默认的优化器类（如果在跑 DeepSpeed，这里会自动变成 DeepSpeedCPUAdam 等）
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            
            # 6. 正式实例化并挂载
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
            
        return self.optimizer