import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def read_data(current_path, data_type, track, language):
    data_path = current_path / 'data' / data_type / track / f'{language}.csv'
    return pd.read_csv(data_path)

# @title Class to handle raw data
class RawData(Dataset):
    def __init__(self, data, labels, tokenizer, max_token):
        self.data = data
        self.labels = labels
        self.max_token = max_token
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        id = self.data['id'].iloc[idx]
        text = self.data['text'].iloc[idx]

        tokenizer_output = self.tokenizer(text, padding='max_length',
                                        truncation=True, max_length=self.max_token, return_tensors='pt')

        input_ids = tokenizer_output.input_ids[0]
        attention_mask = tokenizer_output.attention_mask[0]

        labels = torch.Tensor(self.data[self.labels].iloc[idx].values)

        data = {'id': id,
                'text': text,
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels}
        return data