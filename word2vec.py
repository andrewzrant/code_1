
# coding: utf-8

# Copyright (C) 2016 - 2019 Pinard Liu(liujianping-ok@163.com)
# 
# https://www.cnblogs.com/pinard
# 
# Permission given to modify the code as long as you keep this declaration at the top
# 
# 用gensim学习word2vec https://www.cnblogs.com/pinard/p/7278324.html

# In[3]:


# -*- coding: utf-8 -*-

import jieba
import jieba.analyse

jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('田国富', True)
jieba.suggest_freq('高育良', True)
jieba.suggest_freq('侯亮平', True)
jieba.suggest_freq('钟小艾', True)
jieba.suggest_freq('陈岩石', True)
jieba.suggest_freq('欧阳菁', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('蔡成功', True)
jieba.suggest_freq('孙连城', True)
jieba.suggest_freq('季昌明', True)
jieba.suggest_freq('丁义珍', True)
jieba.suggest_freq('郑西坡', True)
jieba.suggest_freq('赵东来', True)
jieba.suggest_freq('高小琴', True)
jieba.suggest_freq('赵瑞龙', True)
jieba.suggest_freq('林华华', True)
jieba.suggest_freq('陆亦可', True)
jieba.suggest_freq('刘新建', True)
jieba.suggest_freq('刘庆祝', True)

with open('./in_the_name_of_people.txt') as f:
    document = f.read()
    
    #document_decode = document.decode('GBK')
    
    document_cut = jieba.cut(document)
    #print  ' '.join(jieba_cut)  //如果打印结果，则分词效果消失，后面的result无法显示
    result = ' '.join(document_cut)
    result = result.encode('utf-8')
    with open('./in_the_name_of_people_segment.txt', 'w') as f2:
        f2.write(result)
f.close()
f2.close()


# In[64]:


# import modules & set up logging
import logging
import os
from gensim.models import word2vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

sentences = word2vec.LineSentence('./in_the_name_of_people_segment.txt') 

model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100)  


# In[65]:


req_count = 5
for key in model.wv.similar_by_word('李达康'.decode('utf-8'), topn =100):
    if len(key[0])==3:
        req_count -= 1
        print key[0], key[1]
        if req_count == 0:
            break;


# In[66]:


req_count = 5
for key in model.wv.similar_by_word('赵东来'.decode('utf-8'), topn =100):
    if len(key[0])==3:
        req_count -= 1
        print key[0], key[1]
        if req_count == 0:
            break;


# In[67]:


req_count = 5
for key in model.wv.similar_by_word('高育良'.decode('utf-8'), topn =100):
    if len(key[0])==3:
        req_count -= 1
        print key[0], key[1]
        if req_count == 0:
            break;


# In[63]:


req_count = 5
for key in model.wv.similar_by_word('沙瑞金'.decode('utf-8'), topn =100):
    if len(key[0])==3:
        req_count -= 1
        print key[0], key[1]
        if req_count == 0:
            break;


# In[77]:


print model.wv.similarity('沙瑞金'.decode('utf-8'), '高育良'.decode('utf-8'))
print model.wv.similarity('李达康'.decode('utf-8'), '王大路'.decode('utf-8'))


# In[76]:


print model.wv.doesnt_match(u"沙瑞金 高育良 李达康 刘庆祝".split())


# In[ ]:




