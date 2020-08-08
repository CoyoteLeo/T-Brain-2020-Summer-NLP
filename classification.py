import numpy as np
import torch
from transformers import BertForSequenceClassification, BertTokenizer

device = torch.device("cuda")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", do_lower_case=True)
bert_models = []
for model_num in range(1, 6):
    bert_model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
    bert_model.load_state_dict(torch.load(f"model/possible_model{model_num}.pkl"))
    bert_model.eval()
    bert_model.to(device)
    bert_models.append(bert_model)


def predict_is_possible(article):
    score = 0
    encoded_dict = bert_tokenizer.encode_plus(
        article,
        add_special_tokens=True,
        max_length=512,
        truncation=True,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    for bert_model in bert_models:
        input_ids, attention_mask = (
            encoded_dict["input_ids"].to(device),
            encoded_dict["attention_mask"].to(device),
        )
        with torch.no_grad():
            output = bert_model(
                input_ids, token_type_ids=None, attention_mask=attention_mask
            )
            pred = output[0]
            pred = pred.detach().cpu().numpy()
        result = np.argmax(pred)
        score += result
        if score >= 2:
            return True
    return score == 1
