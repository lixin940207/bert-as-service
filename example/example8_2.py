#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Han Xiao <artex.xh@gmail.com> <https://hanxiao.github.io>

# NOTE: First install bert-as-service via
# $
# $ pip install bert-serving-server
# $ pip install bert-serving-client
# $

# simple similarity search on FAQ

import numpy as np
from bert_serving.client import BertClient
from termcolor import colored

prefix_q = '##### **Q:** '
topk = 5

with open('../README.md') as fp:
    # questions = [v.replace(prefix_q, '').strip() for v in fp if v.strip() and v.startswith(prefix_q)]
    questions = [u"为什么自助注册时报错误代码",
                 u"什么人可以申请使用ICBC个人网上银行",
                 u"是否可将本人的网上银行用户名（登录卡号）和密码提供给他人或银行",
                 u"如何办理ICBC个人网上银行的开户手续",
                 u"个人客户使用我行网上银行需要具备怎样的计算机软硬件条件",
                 u"什么是个人网上银行的登录卡和支付卡",
                 u"什么是登录密码和支付密码",
                 u"个人客户的登录密码被锁了怎么办",
                 u"个人客户登录密码忘记或者遗失怎么办",
                 u'客户查询B2C在线支付的交易状态时，系统提示"付款成功，未通知商户"是什么意思',
                 u"客户如何实现B to C在线支付",
                 u"个人客户办理网上挂失后应注意什么问题",
                 u"页面出现“???????”字符怎么办",
                 u"使用个人网上银行应该注意什么"]
    print('%d questions loaded, avg. len of %d' % (len(questions), np.mean([len(d.split()) for d in questions])))

with BertClient(port=5557, port_out=5558, ignore_all_checks=True) as bc:
    doc_vecs = bc.encode(questions)

    while True:
        query = input(colored('your question: ', 'green'))
        # query = str(query)
        query_vec = bc.encode([query])[0]
        # compute normalized dot product as score
        score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
        topk_idx = np.argsort(score)[::-1][:topk]
        print('top %d questions similar to "%s"' % (topk, colored(query, 'green')))
        for idx in topk_idx:
            print(u'> %s\t%s' % (colored('%.1f' % score[idx], 'cyan'), colored(questions[idx], 'yellow')))
            # print(score[idx])
