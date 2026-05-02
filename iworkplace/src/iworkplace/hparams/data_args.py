from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    
    dataset_name: str = field(default="drop", metadata={"help": "The name of the dataset to use."})
    train_data_path: str = field(default="./data/train.json", metadata={"help": "Path to the training data."})
    test_data_path: str = field(default="./data/test.json", metadata={"help": "Path to the testing/eval data."})
    
    max_prompt_length: int = field(default=4096, metadata={"help": "Maximum sequence length for the prompt."})
    max_qa_pair: int = field(default=10, metadata={"help": "Maximum number of QA pairs per training sample."})
    infer_max_qa_pair: int = field(default=10, metadata={"help": "Maximum number of QA pairs per inference sample."})
    cond_max_length: Optional[int] = field(default=None, metadata={"help": "Maximum length for condition embeddings."})
    
    # Knowledge & Concepts
    use_concepts: bool = field(default=False, metadata={"help": "Whether to use concept grounding."})
    num_concepts: Optional[int] = field(default=None, metadata={"help": "Number of concepts to use."})
    use_knowledge_graph: bool = field(default=False, metadata={"help": "Whether to incorporate knowledge graph data."})