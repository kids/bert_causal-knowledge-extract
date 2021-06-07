# -*- coding: utf-8 -*-

import torch
from transformers import BertTokenizer, BertModel
import os
import random
import logging


class SCls(torch.nn.Module):
    def __init__(self, input_dim, dropout_rate=0.):
        super(SCls, self).__init__()
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.linear = torch.nn.Linear(input_dim, 2)

    def forward(self, x):
        x = self.dropout(x)
        y = self.linear(x)
        return y


def semb(mdl, tokenizer, s):
    s = tokenizer(s)
    tok_emb, s_emb = mdl(torch.tensor([s['input_ids']]))
    return s_emb[0]


def mk_data(f, tokenizer, b_mdl):
    with open(f) as fr:
        txt = fr.read()
    train_dt = []
    for l in txt.split('\n')):
        if len(i.strip()) < 5:
            continue
        try:
            s, l=i.strip().split(',')
            train_dt.append((semb(s), int(l)))
        except:
            logging.warming('Sample format error: ' + i)
            continue
    return train_dt


def train():
    train_dt_pos = [i for i in train_dt if i[1] == 1]
    train_dt_neg = [i for i in train_dt if i[1] == 0]
    logging.debug('sample size(pos,neg):{},{}'.format(len(train_dt_pos), len(train_dt_neg)))
    batch_size = 10
    pos_c_max = 7
    lr = 0.02
    tt_loss = 0
    optimizer = torch.optim.SGD(scls.parameters(), lr = lr)
    loss = torch.nn.CrossEntropyLoss()
    global_step = 0
    scls.train()
    for _ in range(50):
        pos_c = 4  # random.randint(1,pos_c_max)
        batch_dt = random.sample(train_dt_pos, pos_c) + random.sample(train_dt_neg, batch_size - pos_c)
        if type(batch_dt[0][0]) == torch.Tensor:
            batch_input = torch.cat([i[0] for i in batch_dt]).view(batch_size,-1)
        else:
            batch_input = torch.tensor([i[0] for i in batch_dt])
        yp = scls(batch_input)
        y = [i[1] for i in batch_dt[:batch_size]]
        logging.debug('yp vs y:{},{}'.format(yp, y))
        l = loss(yp, torch.tensor(y))
        l.backward()
        tt_loss += l.item()
        torch.nn.utils.clip_grad_norm_(scls.parameters(), 1.0)
        optimizer.step()
        scls.zero_grad()
        global_step += 1
        logging.debug('{}'.format(tt_loss / global_step))
    torch.save(scls, './scls')


def infer():
    scls.eval()
    for s in '''
    只需2万 就能变成“人才” 买房可赚百万差价！房产中介：我们为留住人才“操碎了心”
    图解陆股通：55家公司获北向资金明显增持 这些股最活跃
    监管现场检查拟IPO企业 5大问题细致入微 年内撤回申请企业数超去年全年
    中科院叶培建院士：中国小天体探测任务已进入工程研制阶段
    腾讯公开“社交账号单向好友检测”相关专利 无需群发冗余信息
    “炒鞋”被骗137万！行骗者是“95后炒鞋大佬” 获刑10年半 庭审回放曝光详情
    中信证券：库存拐点、供应紧张、美元下行成为铜价中期核心驱动力'''.strip().split('\n')[:]:
        x=semb(s.strip().split(',')[0])
        yp=scls(torch.tensor([x]))
        pos=int(yp[0][1] > .5)
        if pos or (',' in s and s.split(',')[1] == '1'):
            logging.debug('{},{},{}'.format(s.strip(), pos, yp.detach()))


if __name__ == '__main__':
    mdl_path='~/Downloads/tmp1/mdl_bert/'
    if not os.path.exist(mdl_path):
        mdl_path='bert-base-chinese'
    tokenizer=BertTokenizer.from_pretrained(mdl_path)
    mdl=BertModel.from_pretrained(mdl_path)

    scls=SCls(768)
    if os.path.exist('./scls')
        scls=torch.load('./scls')
