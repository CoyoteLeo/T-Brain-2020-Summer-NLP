#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import re
import pickle
import ast
from zhon.hanzi import non_stops, stops
import torch
import torch.nn.functional as F
from transformers import RobertaTokenizer , RobertaModel , RobertaForSequenceClassification
from transformers import BertTokenizer , BertConfig , BertModel ,BertForSequenceClassification
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import nn


# In[14]:


def eval(pred, ans):
    if bool(pred) is not bool(ans):
        return 0
    elif not pred and not ans:
        return 1
    else:
        pred = set(pred)
        ans = set(ans)
        interaction_len = len(pred & ans)
        if interaction_len == 0:
            return 0

        pred_len = len(pred)
        ans_len = len(ans)
        return 2 / (pred_len / interaction_len + ans_len / interaction_len)


def eval_all(pred_list, ans_list):
    assert len(pred_list) == len(ans_list)
    return sum(eval(p, a) for p, a in zip(pred_list, ans_list)) / len(pred_list)


# In[15]:


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

def cut_sent(para):
    para = re.sub("([。！？\?])([^”’])", r"\1\n\2", para)
    para = re.sub("(\.{6})([^”’])", r"\1\n\2", para)
    para = re.sub("(\…{2})([^”’])", r"\1\n\2", para)
    para = re.sub("([。！？\?][”’])([^，。！？\?])", r"\1\n\2", para)
    return para.split("\n")

def clean_string(content):
#     cc = OpenCC('t2s')
    content = content.replace('\n','。').replace('\t','，').replace('!', '！').replace('?', '？')# erease white space cause English name error
    content = re.sub("[+\.\/_,$%●▼►^*(+\"\']+|[+——~@#￥%……&*（）★]", "",content)
    content = re.sub(r"[%s]+" %stops, "。",content)
#     content = cc.convert(content)
    return content


# In[16]:



def qa_binary_split_data(df):
    tokenizer = BertTokenizer.from_pretrained(lm_path)
    # tokenizer = RobertaTokenizer.from_pretrained(lm_path)


    train_x = []
    train_y = []
    train_input_ids = []
    train_token_types = []
    train_attention_mask = []


    for index , row in df.iterrows():
        news = row['full_content']
        ckip_names = ast.literal_eval(row['ckip_names'])
        names  = ast.literal_eval(row['name'])

        if len(names) == 0 :
            continue

        content = clean_string(news)
        max_length = 500

        split_content = cut_sent(content)
        chunks = combine_sentence(split_content, max_length)

        for chunk in chunks:
            for ckip_name in ckip_names:
                if len(chunk) >= max_length:
                    print("error !!!! lenth > 500")
                    continue
                if ckip_name not in chunk:
                    continue

                input_ids = tokenizer.encode(ckip_name, chunk)
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
                
                if ckip_name in names:
                    train_y.append(1)
                else:
                    train_y.append(0)

                train_input_ids.append(input_ids)
                train_token_types.append(segment_ids)
                train_attention_mask.append(input_mask)
                train_x.append((ckip_name,chunk))

    train_input_ids = np.array(train_input_ids)
    train_token_types = np.array(train_token_types)
    train_attention_mask = np.array(train_attention_mask)
    train_y = np.array(train_y)

    print(len(train_x))
    print(train_input_ids.shape)
    print(train_token_types.shape)
    print(train_attention_mask.shape)
    print(train_y.shape)

    return train_x , train_input_ids , train_token_types , train_attention_mask , train_y
                
                


# In[17]:


class TrainDataset(Dataset):
    def __init__(self, input_ids, token_type_ids, attention_mask, y , x):
        self.input_ids = input_ids
        self.token_type_ids = token_type_ids
        self.attention_mask = attention_mask
        self.y = y
        self.x = x

    def __getitem__(self, idx):
        inputid = self.input_ids[idx]
        tokentype = self.token_type_ids[idx]
        attentionmask = self.attention_mask[idx]
        x = self.x[idx]
        y = self.y[idx]
        return inputid, tokentype, attentionmask,  y , x

    def __len__(self):
        return len(self.input_ids)


# In[18]:


def get_test_acc(model , dataloader):

    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        for data in dataloader:
            tokens_tensors ,  segments_tensors , masks_tensors , labels = [t.to(device) for t in data[:-1]]

            name , chunk = data[-1]

            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors,
                            labels = labels)
            
            pred = outputs[1]

            total += len(tokens_tensors)
            pred = torch.argmax(pred,dim=-1)

            correct += (pred == labels).sum().item()
    return correct/total



# In[7]:


# dataset = 1
for dataset in range(1,4):
    dataset_base_path = './dataset/dataset'


    lm_path = './chinese_roberta_wwm/'
    train_df = pd.read_csv(dataset_base_path + str(dataset) + '/tbrain_train.csv')
    test_df = pd.read_csv(dataset_base_path + str(dataset) + '/tbrain_test.csv')

    print(train_df.shape)
    print(test_df.shape)


    train_x , train_input_ids , train_token_types , train_attention_mask , train_y  = qa_binary_split_data(train_df)
    test_x , test_input_ids , test_token_types , test_attention_mask , test_y  = qa_binary_split_data(test_df)


    from transformers import BertForSequenceClassification
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print("device:", device)
    print('dataset', dataset)

    num_labels = 2

    model = BertForSequenceClassification.from_pretrained(lm_path,num_labels=num_labels)
    model.to(device)
#     model.bert.init_weights()
    model.train()

    BATCH_SIZE = 10
    trainset = TrainDataset(
        train_input_ids, train_token_types, train_attention_mask, train_y,train_x
    )
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True)

    testset = TrainDataset(
        test_input_ids, test_token_types, test_attention_mask, test_y ,test_x
    )
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

    EPOCHS = 4
    for epoch in range(EPOCHS):
        running_loss = 0.0
        total = 0
        correct = 0
        for data in trainloader:
            tokens_tensors ,  segments_tensors , masks_tensors , labels = [t.to(device) for t in data[:-1]]

            name , chunk = data[-1]

            optimizer.zero_grad()
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors,
                            labels = labels)
            
            loss = outputs[0]
            pred = outputs[1]

            total += len(tokens_tensors)
            pred = torch.argmax(pred,dim=-1)

            correct += (pred == labels).sum().item()


            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            
        

        print('epoch:',epoch)
        print('loss:', running_loss)
        print('train_acc:',correct/total)
        print('test_acc:',get_test_acc(model,testloader))
        checkpoint_path = './QAModel/' + str(dataset) + '/roberta_init2_name_qa_split_epoch' + str(epoch) + '.pkl'
        torch.save(model.state_dict(),checkpoint_path)
    print('=====================================')


# In[19]:


def check_pred_name_is_real_ans(pred_name_list, news , checkpoint , lm_path):
    num_labels = 2
    model = BertForSequenceClassification.from_pretrained(lm_path,num_labels=num_labels)
    model.load_state_dict(torch.load(checkpoint))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    tokenizer = BertTokenizer.from_pretrained(lm_path)

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
                print("error !!!! lenth > 500")
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
            outputs = model(
                input_ids=tokens_tensors,
                token_type_ids=segments_tensors,
                attention_mask=masks_tensors,
            )
            pred = torch.softmax(outputs[0], dim=-1)
            pred = torch.argmax(pred, dim=-1)
            pred = pred.cpu().detach().numpy()
            pred_name_list = np.array(name)
            return list(pred_name_list[pred > 0])


# In[21]:


dataset = 3
dataset_base_path = './dataset/dataset'

model_path = './QAModel/'+ str(dataset) +'/'


# rbt_checkpoint = model_path +  'roberta_name_qa_split_epoch1.pkl'
# # rbtl3_checkpoint = model_path + 'rbtl3_name_qa_split_epoch0.pkl'
# # bert_checkpoint = model_path + 'bert_name_qa_split_epoch2.pkl'

rbt0_checkpoint = model_path + 'roberta_init0_name_qa_split_epoch2.pkl'
rbt1_checkpoint = model_path + 'roberta_init1_name_qa_split_epoch2.pkl'
rbt2_checkpoint = model_path + 'roberta_init2_name_qa_split_epoch1.pkl'





rbt_lm_path = './chinese_roberta_wwm/'
# rbtl3_lm_path = './rbtl3_pretrain'
# bert_lm_path = './bert_wwm_pretrain_tbrain/'


train_df = pd.read_csv(dataset_base_path + str(dataset) + '/tbrain_train.csv')
test_df = pd.read_csv(dataset_base_path + str(dataset) + '/tbrain_test.csv')

print(train_df.shape)
print(test_df.shape)


ans = []
rbt0_pred = []
rbt1_pred = []
rbt2_pred = []
vote_pred = []
vote_all_pred = []
for index,row in test_df.iterrows():
    news = row['full_content']
    ckip_names = ast.literal_eval(row['ckip_names'])
    names = ast.literal_eval(row['name'])


    if len(names) == 0:
        continue

    ans.append(names)


    rbt0_result = check_pred_name_is_real_ans(ckip_names, news , rbt0_checkpoint , rbt_lm_path)
    rbt0_result = list(set(rbt0_result))
    rbt0_pred.append(rbt0_result)
    
    rbt1_result = check_pred_name_is_real_ans(ckip_names, news , rbt1_checkpoint , rbt_lm_path)
    rbt1_result = list(set(rbt1_result))
    rbt1_pred.append(rbt1_result)
    
    rbt2_result = check_pred_name_is_real_ans(ckip_names, news , rbt2_checkpoint , rbt_lm_path)
    rbt2_result = list(set(rbt2_result))
    rbt2_pred.append(rbt2_result)




    tmp = []
    tmp_all = []
    for name in list(set(rbt0_result + rbt1_result + rbt2_result )):
        vote = 0
        vote += name in rbt0_result
        vote += name in rbt1_result
        vote += name in rbt2_result
        if vote >=2:
            tmp.append(name)
        if vote == 3:
            tmp_all.append(name)
            
    vote_pred.append(tmp)
    vote_all_pred.append(tmp_all)
    
    print('------------')
    print('ans:',names)
    print('vote:',tmp)
    print('vote_all:',tmp_all)
            





    # ensemble_or.append(rbtl3_result or rbt_result)
    # ensemble_and.append(rbtl3_result and rbt_result)


print('dataset:',dataset)
print('rbt0: %.4f' % eval_all(rbt0_pred,ans))
print('rbt1: %.4f' % eval_all(rbt1_pred,ans))
print('rbt2: %.4f' % eval_all(rbt2_pred,ans))

# print('rbtl3:',eval_all(rbtl3_pred,ans))
# print('bert:',eval_all(bert_pred,ans))
print('vote: %.4f' % eval_all(vote_pred,ans))
print('vote_all: %.4f' % eval_all(vote_all_pred,ans))
# print('or:',eval_all(ensemble_or,ans))
# print('and:',eval_all(ensemble_and,ans))


# In[17]:


print('vote_all: %.4f' % eval_all(['2'],['123']))


# In[20]:


def qa_name_binary_ensemble(pred_name_list, news):
    num_labels = 2
    lm_path = './chinese_roberta_wwm/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_path = './QAModel/1/'

    rbt0_checkpoint = model_path + 'roberta_init0_all_data_name_qa_split_epoch0.pkl'
    rbt1_checkpoint = model_path + 'roberta_init1_all_data_name_qa_split_epoch2.pkl'
    rbt2_checkpoint = model_path + 'roberta_init2_all_data_name_qa_split_epoch2.pkl'

    
    
    model0 = BertForSequenceClassification.from_pretrained(lm_path,num_labels=num_labels)
    model0.load_state_dict(torch.load(rbt0_checkpoint))
    model0.to(device)
    model0.eval()
    
    model1 = BertForSequenceClassification.from_pretrained(lm_path,num_labels=num_labels)
    model1.load_state_dict(torch.load(rbt1_checkpoint))
    model1.to(device)
    model1.eval()
    
    model2 = BertForSequenceClassification.from_pretrained(lm_path,num_labels=num_labels)
    model2.load_state_dict(torch.load(rbt2_checkpoint))
    model2.to(device)
    model2.eval()
    
    
    

    tokenizer = BertTokenizer.from_pretrained(lm_path)

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
                print("error !!!! lenth > 500")
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
            
            outputs0 = model0(
                input_ids=tokens_tensors,
                token_type_ids=segments_tensors,
                attention_mask=masks_tensors,
            )
            pred0 = torch.softmax(outputs0[0], dim=-1)
            pred0 = torch.argmax(pred0, dim=-1)
            pred0 = pred0.cpu().detach().numpy()
            ans0 = list(pred_name_list[pred0 > 0])
            
            outputs1 = model1(
                input_ids=tokens_tensors,
                token_type_ids=segments_tensors,
                attention_mask=masks_tensors,
            )
            pred1 = torch.softmax(outputs1[0], dim=-1)
            pred1 = torch.argmax(pred1, dim=-1)
            pred1 = pred1.cpu().detach().numpy()
            ans1 = list(pred_name_list[pred1 > 0])
            
            
            outputs2 = model2(
                input_ids=tokens_tensors,
                token_type_ids=segments_tensors,
                attention_mask=masks_tensors,
            )
            pred2 = torch.softmax(outputs2[0], dim=-1)
            pred2 = torch.argmax(pred2, dim=-1)
            pred2 = pred2.cpu().detach().numpy()
            ans2 = list(pred_name_list[pred2 > 0])
            
            
            vote_result = []
            for name in list(set(ans0 + ans1 + ans2)):
                vote = 0
                vote += name in ans0
                vote += name in ans1
                vote += name in ans2
                if vote >=2:
                    vote_result.append(name)

            
            return vote_result


# In[22]:



# 這邊是每天的 validation csv 輸出code
import pandas as pd

test_df = pd.read_csv('./tbrain/2020-07-27.csv')
validation_df =  pd.DataFrame(columns=['idx', 'article','ckip_name' , 'original_output' , 'only_QA_output'])

count = 0
for index, row in test_df.iterrows():
    if(row['binary'] != 1):
        continue

    news = row['article']
    ckip_name = ast.literal_eval(row['ckip_name'])
    pred_name_list = ast.literal_eval(row['predict_name'])
    pred_name_list = sorted(list(set(pred_name_list)))
    
    only_qa_pred = qa_name_binary_ensemble(ckip_name,news)
    only_qa_pred = sorted(list(set(only_qa_pred)))

    validation_df.loc[count] = [str(index), news , str(ckip_name) ,  str(pred_name_list) , str(only_qa_pred) ]
    count += 1

validation_df.to_csv('./tbrain/2020-07-27_after_ensemble.csv',index=False)
    


# In[7]:


dataset = 1

dataset_base_path = './dataset/dataset'


lm_path = './chinese_roberta_wwm/'
train_df = pd.read_csv(dataset_base_path + str(dataset) + '/tbrain_train.csv')
test_df = pd.read_csv(dataset_base_path + str(dataset) + '/tbrain_test.csv')

train_df = pd.concat([train_df,test_df])

print(train_df.shape)
print(test_df.shape)


train_x , train_input_ids , train_token_types , train_attention_mask , train_y  = qa_binary_split_data(train_df)
# test_x , test_input_ids , test_token_types , test_attention_mask , test_y  = qa_binary_split_data(test_df)




from transformers import BertForSequenceClassification
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print("device:", device)
print('dataset', dataset)

num_labels = 2

model = BertForSequenceClassification.from_pretrained(lm_path,num_labels=num_labels)
model.to(device)
model_path = './QAModel/'+ str(dataset) +'/'
checkpoint = model_path + 'roberta_init2_name_qa_split_epoch1.pkl'
model.load_state_dict(torch.load(checkpoint))
#     model.bert.init_weights()
model.train()

BATCH_SIZE = 10
trainset = TrainDataset(
    train_input_ids, train_token_types, train_attention_mask, train_y,train_x
)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE,shuffle=True)

# testset = TrainDataset(
#     test_input_ids, test_token_types, test_attention_mask, test_y ,test_x
# )
# testloader = DataLoader(testset, batch_size=BATCH_SIZE,shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

EPOCHS = 4
for epoch in range(EPOCHS):
    running_loss = 0.0
    total = 0
    correct = 0
    for data in trainloader:
        tokens_tensors ,  segments_tensors , masks_tensors , labels = [t.to(device) for t in data[:-1]]

        name , chunk = data[-1]

        optimizer.zero_grad()
        outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors,
                        labels = labels)

        loss = outputs[0]
        pred = outputs[1]

        total += len(tokens_tensors)
        pred = torch.argmax(pred,dim=-1)

        correct += (pred == labels).sum().item()


        running_loss += loss.item()
        loss.backward()
        optimizer.step()



    print('epoch:',epoch)
    print('loss:', running_loss)
    print('train_acc:',correct/total)
#     print('test_acc:',get_test_acc(model,testloader))
    checkpoint_path = './QAModel/' + str(dataset) + '/roberta_init2_all_data_name_qa_split_epoch' + str(epoch) + '.pkl'
    torch.save(model.state_dict(),checkpoint_path)
print('=====================================')


# In[ ]:




