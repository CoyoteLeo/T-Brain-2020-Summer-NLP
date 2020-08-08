import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import BertForSequenceClassification, BertTokenizer

from utils import clean_string, cut_sent

num_labels = 2
lm_path = "model/chinese_roberta_wwm/"
device = torch.device("cuda")


models = []
for model_num in range(1, 4):
    model = BertForSequenceClassification.from_pretrained(lm_path, num_labels=2)
    model.load_state_dict(torch.load(f"model/name_model{model_num}.pkl"))
    model.to(device)
    model.eval()
    models.append(model)

tokenizer = BertTokenizer.from_pretrained(lm_path)


def validate_name(pred_name_list, news):
    class Testset(Dataset):
        def __init__(self, input_ids, token_type_ids, attention_mask, names):
            self.input_ids = input_ids
            self.token_type_ids = token_type_ids
            self.attention_mask = attention_mask
            self.names = names

        def __getitem__(self, idx):
            inputid = self.input_ids[idx]
            tokentype = self.token_type_ids[idx]
            attentionmask = self.attention_mask[idx]
            name = self.names[idx]
            return inputid, tokentype, attentionmask, name

        def __len__(self):
            return len(self.input_ids)

    def combine_sentence(sentences, max_len):
        li = []
        string = ""
        for k in range(len(sentences)):
            sentence = sentences[k]
            if len(string) + len(sentence) < max_len:
                string = string + sentence
            else:
                #             原本是空的代表sentences太常
                if string == "":
                    n = max_len
                    tmp_li = [sentence[i : i + n] for i in range(0, len(sentence), n)]
                    string = tmp_li.pop(-1)
                    li = li + tmp_li
                else:
                    li.append(string)
                    string = sentence
        if string != "":
            li.append(string)
        return li

    train_input_ids = []
    train_token_types = []
    train_attention_mask = []
    testing_name = []

    content = clean_string(news)

    max_length = 500

    split_content = cut_sent(content)
    chunks = combine_sentence(split_content, max_length)

    for chunk in chunks:
        for name in pred_name_list:
            if len(chunk) >= max_length:
                continue
            if name not in chunk:
                continue

            input_ids = tokenizer.encode(name, chunk)
            if len(input_ids) > 512:
                continue
            sep_index = input_ids.index(tokenizer.sep_token_id)
            num_seg_a = sep_index + 1
            num_seg_b = len(input_ids) - num_seg_a
            segment_ids = [0] * num_seg_a + [1] * num_seg_b

            input_mask = [1] * len(input_ids)

            while len(input_ids) < 512:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            train_input_ids.append(input_ids)
            train_token_types.append(segment_ids)
            train_attention_mask.append(input_mask)
            testing_name.append(name)

    train_input_ids = np.array(train_input_ids)
    train_token_types = np.array(train_token_types)
    train_attention_mask = np.array(train_attention_mask)
    testing_name = np.array(testing_name)

    BATCH_SIZE = train_input_ids.shape[0]
    testset = Testset(
        train_input_ids, train_token_types, train_attention_mask, testing_name
    )
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)

    with torch.no_grad():
        for data in testloader:
            tokens_tensors, segments_tensors, masks_tensors = [
                t.to(device) for t in data[:-1]
            ]
            name = data[-1]
            pred_name_list = np.array(name)

            ans = []
            for model in models:
                outputs = model(
                    input_ids=tokens_tensors,
                    token_type_ids=segments_tensors,
                    attention_mask=masks_tensors,
                )
                pred = torch.softmax(outputs[0], dim=-1)
                pred = torch.argmax(pred, dim=-1)
                pred = pred.cpu().detach().numpy()
                ans.append(list(pred_name_list[pred > 0]))

            vote_result = []
            for name in set(itertools.chain.from_iterable(ans)):
                vote = 0
                vote += name in ans[0]
                vote += name in ans[1]
                vote += name in ans[2]
                if vote >= 2:
                    vote_result.append(name)

            return vote_result
