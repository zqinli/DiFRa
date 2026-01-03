from typing import Union, Optional
from trl import SFTTrainer
from datasets import Dataset, DatasetDict, concatenate_datasets
import torch

class QAGTrainer(SFTTrainer):
    
    def _prepare_dataset(
        self,
        dataset: Union[Dataset, DatasetDict],
        processing_class,
        training_args,
        packing: bool = False,
        formatting_function=None,
        dataset_name: Optional[str] = None,
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
        if hasattr(self.model, "log_vars"):
            with torch.no_grad():
                log_vars = self.model.log_vars.detach().cpu()
                
                weight_main = (0.5 * torch.exp(-log_vars[0])).item()
                weight_diff = (0.5 * torch.exp(-log_vars[1])).item()

                logs["weight/main"] = weight_main
                logs["weight/diff"] = weight_diff
                
                logs["sigma/main"] = torch.exp(0.5 * log_vars[0]).item()
                logs["sigma/diff"] = torch.exp(0.5 * log_vars[1]).item()

        super().log(logs, *args, **kwargs)