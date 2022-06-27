import torch.nn
from transformers import BertModel, BertTokenizer
import jieba
from torchDL.snippets import extract_matching, sequence_padding
import json
import numpy as np

pretrain_model_path = r"F:/Model/bert_base_chinese/"
model = BertModel.from_pretrained(pretrain_model_path)
tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)


def token_encode(text):
    token_ids = tokenizer.encode(text)
    return token_ids


def load_data(path):
    batch_token, batch_seg, batch_mask, batch_label, batch_cls = [], [], [], [], []
    for data in json.load(open(path, 'r', encoding="utf-8")):
        summary, content = data["summary"], data["content"]
        summary_lst = jieba.lcut(summary)
        content_lst = jieba.lcut(content)
        idx_map = extract_matching(content_lst, summary_lst)
        idx_lst = [i[1] for i in idx_map]
        token_ids = []
        label_lst = []
        cls_lst = []
        for idx, i in enumerate(content_lst):
            tt = token_encode(i)
            lt = [0] * len(tt)
            if idx in idx_lst:
                lt[0] = 1
            clst = [0] * len(tt)
            clst[0] = 1
            label_lst.extend(lt)
            token_ids.extend(tt)
            cls_lst.append(clst)
        cls_lst = [idx for idx, i in enumerate(cls_lst) if i]
        mask_lst = [1] * len(label_lst)

        batch_token.append(np.array(token_ids))
        batch_label.append(np.array(label_lst))
        batch_mask.append(np.array(mask_lst))
        batch_seg.append(np.array(mask_lst))
        batch_cls.append(np.array(cls_lst))

        if not len(batch_token) % 10:
            batch_token = sequence_padding(batch_token)
            batch_seg = sequence_padding(batch_seg)
            batch_mask = sequence_padding(batch_mask)
            batch_label = sequence_padding(batch_label)
            # batch_cls = sequence_padding(batch_cls)
            yield batch_token, batch_seg, batch_mask, batch_label, batch_cls
            batch_token, batch_seg, batch_mask, batch_label, batch_cls = [], [], [], [], []


loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-5)


def train(epoch):
    for ep in range(epoch):
        count = 0
        for data in load_data("../dataset/lcsts/lcsts_train.json"):
            count += 1
            logist = model.forward(input_ids=torch.tensor(data[0]), attention_mask=torch.tensor(data[2]),
                                   token_type_ids=torch.tensor(data[1]))["last_hidden_state"]
            loss = 0
            for i in range(len(data[4])):
                output = logist[i, data[4][i]]
                # output = torch.unsqueeze(output, 0)
                label = torch.tensor(data[3][i, data[4][i]], dtype=torch.long)
                loss += loss_fn(output, label)
            optim.zero_grad()
            loss.backward()
            optim.step()

            if not count % 200:
                with torch.no_grad():
                    dev_loss = 0
                    count1 = 0
                    for dev_data in load_data("../dataset/lcsts/lcsts_dev.json"):
                        count1 += 1
                        logist = model.forward(input_ids=torch.tensor(dev_data[0]), attention_mask=torch.tensor(dev_data[2]),
                                   token_type_ids=torch.tensor(dev_data[1]))["last_hidden_state"]
                        loss = 0
                        for i in range(len(data[4])):
                            output = logist[i, data[4][i]]
                            # output = torch.unsqueeze(output, 0)
                            label = torch.tensor(data[3][i, data[4][i]], dtype=torch.long)
                            loss += loss_fn(output, label)
                        dev_loss += loss
                    print(f"dev_loss:{dev_loss/count1}")
        print(f"save model:model_{ep}.pt")
        torch.save(model.state_dict(), f"model_{ep}.pt")
train(5)