import torch.nn
from transformers import BertModel, BertTokenizer
import jieba
from torchDL.snippets import extract_matching, sequence_padding
import json

pretrain_model_path = r"F:/Model/bert_base_chinese/"
model = BertModel.from_pretrained(pretrain_model_path)
tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)


def token_encode(text):
    token_ids = tokenizer.encode(text)
    return token_ids


def load_data(path):
    batch_token, batch_seg, batch_mask, batch_label = [], [], [], []
    for data in json.load(open(path, 'r', encoding="utf-8")):
        summary, content = data["summary"], data["content"]
        summary_lst = jieba.lcut(summary)
        content_lst = jieba.lcut(content)
        idx_map = extract_matching(content_lst, summary_lst)
        idx_lst = [i[1] for i in idx_map]
        token_ids = []
        label_lst = []
        for idx, i in enumerate(content_lst):
            tt = token_encode(i)
            lt = [0] * len(tt)
            if idx in idx_lst:
                lt[0] = 1
            label_lst.extend(lt)
            token_ids.extend(tt)
        mask_lst = [1] * len(label_lst)

        batch_token.append(torch.tensor(token_ids))
        batch_label.append(torch.tensor(label_lst))
        batch_mask.append(torch.tensor(mask_lst))
        batch_seg.append(torch.tensor(mask_lst))

        if not len(batch_token) % 10:
            batch_token = sequence_padding(batch_token)
            batch_seg = sequence_padding(batch_seg)
            batch_mask = sequence_padding(batch_mask)
            batch_label = sequence_padding(batch_label)
            yield batch_token, batch_seg, batch_mask, batch_label
            batch_token, batch_seg, batch_mask, batch_label = [], [], [], []


loss_fn = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=1e-5)


def train(epoch):
    for ep in range(epoch):
        for data in load_data("../dataset/lcsts/lcsts_train.json"):
            logist = model.forward(input_ids=data[0], attention_mask=data[1], token_type_ids=data[2])
            loss = loss_fn(logist, data[3])
            optim.zero_grad()
            loss.backward()
            optim.step()

            with torch.no_grad():
                dev_loss = 0
                count = 0
                for dev_data in load_data("../dataset/lcsts/lcsts_dev.json"):
                    count += 1
                    logist = model.forward(dev_data[0], dev_data[1], dev_data[2])
                    loss = loss_fn(logist, dev_data[3])
                    dev_loss += loss
                print(f"dev_loss:{dev_loss/count}")
        print(f"save model:model_{ep}.pt")
        torch.save(model.state_dict(), f"model_{ep}.pt")
train(5)