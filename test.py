from datasets import load_dataset
from transformers import RobertaTokenizerFast
sample_dataset_raw = load_dataset(
                'wikipedia', '20220301.en'
            )
tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')



from src.data.tok_dataset import TokenizedDataset

sample_dataset = TokenizedDataset(
    sample_dataset_raw['train'], 
    tokenizer, 
    maxlen=512
)
# sample_dataset[19958, 33265, 33783]
sample_dataset[1,2,3]