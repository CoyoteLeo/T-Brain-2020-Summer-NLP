import re

from torch import nn
from torch.utils.data import Dataset


def clean_string(content):
    content = (
        content.replace("\n", "").replace("\t", "").replace(" ", "").replace("\xa0", "")
    )
    content = re.sub("[●▼►★]", "", content)
    return content


def cut_sent(para):
    para = re.sub("([。！？\?])([^”’])", r"\1\n\2", para)
    para = re.sub("(\.{6})([^”’])", r"\1\n\2", para)
    para = re.sub("(\…{2})([^”’])", r"\1\n\2", para)
    para = re.sub("([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    return para.split("\n")


def combine_sentence(sentences):
    li = []
    string = ""
    for sentence_ in sentences:
        sentence = sentence_
        if len(string) + len(sentence) < 510:
            string += sentence
        else:
            #             原本是空的代表sentences太常
            if string == "":
                n = 510
                tmp_li = [sentence[i : i + n] for i in range(0, len(sentence), n)]
                string = tmp_li.pop(-1)
                li += tmp_li
            else:
                li.append(string)
                string = sentence
    if string != "":
        li.append(string)
    return li


class TestDataset(Dataset):
    def __init__(self, input_dict, text):
        self.input_ids = input_dict["input_ids"]
        self.token_type_ids = input_dict["token_type_ids"]
        self.attention_mask = input_dict["attention_mask"]
        self.text = text

    def __getitem__(self, idx):
        inputid = self.input_ids[idx]
        tokentype = self.token_type_ids[idx]
        attentionmask = self.attention_mask[idx]
        text = self.text[idx]

        return inputid, tokentype, attentionmask, text

    def __len__(self):
        return len(self.input_ids)


class posClassfication_new(nn.Module):
    def __init__(self):
        super(posClassfication_new, self).__init__()
        self.start_task = nn.Sequential(nn.Linear(768, 1),)
        self.end_task = nn.Sequential(nn.Linear(768, 1),)
        self.binary_task = nn.Sequential(nn.Linear(768, 2),)

    #
    def forward(self, start_x, end_x, pool_cls):
        start_x = start_x.double()
        end_x = end_x.double()
        pool_cls = pool_cls.double()

        start_out = self.start_task(start_x)
        end_out = self.end_task(end_x)
        binary_out = self.binary_task(pool_cls)

        return start_out, end_out, binary_out
