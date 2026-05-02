from iworkplace.hparams import DataArguments, ModelArguments
from .load_dataset import load_dataset

UNK_TOKEN = "<unk>"

def get_dataset(data_args: DataArguments, model_args: ModelArguments, bert_tokenizer=None):

    if data_args.dataset_name not in ["drop", "squad"]:
        raise ValueError(f"目前仅支持 'drop' 和 'squad' 数据集，当前输入为: {data_args.dataset_name}")
    
    input_path = {
        "train_data_path": data_args.train_data_path,
        "test_data_path": data_args.test_data_path
    }
    
    dataset_train, dataset_test = load_dataset(
        input_path=input_path,
        max_qa_pair=data_args.max_qa_pair,
        infer_max_qa_pair=data_args.infer_max_qa_pair,
        use_concepts=data_args.use_concepts,
        num_concepts=data_args.num_concepts,
        use_diffusion=model_args.use_diffusion,
        unk_token=UNK_TOKEN,
        use_knowledge_graph=data_args.use_knowledge_graph,
        bert_tokenizer=bert_tokenizer,
    )
    
    return dataset_train, dataset_test