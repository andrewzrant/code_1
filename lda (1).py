
# coding: utf-8

# Copyright (C) 2016 - 2019 Pinard Liu(liujianping-ok@163.com)
# 
# https://www.cnblogs.com/pinard
# 
# Permission given to modify the code as long as you keep this declaration at the top
# 
# 用scikit-learn学习LDA主题模型 https://www.cnblogs.com/pinard/p/6908150.html

# In[44]:


# -*- coding: utf-8 -*-

import jieba

with open('./nlp_test0.txt') as f:
    document = f.read()
    
    document_decode = document.decode('GBK')
    document_cut = jieba.cut(document_decode)
    #print  ' '.join(jieba_cut)  //如果打印结果，则分词效果消失，后面的result无法显示
    result = ' '.join(document_cut)
    result = result.encode('utf-8')
    with open('./nlp_test1.txt', 'w') as f2:
        f2.write(result)
f.close()
f2.close()


# In[45]:


jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('京州', True)


# In[46]:


with open('./nlp_test0.txt') as f:
    document = f.read()
    
    document_decode = document.decode('GBK')
    document_cut = jieba.cut(document_decode)
    #print  ' '.join(jieba_cut)
    result = ' '.join(document_cut)
    result = result.encode('utf-8')
    with open('./nlp_test1.txt', 'w') as f2:
        f2.write(result)
f.close()
f2.close()     


# In[47]:


#从文件导入停用词表
stpwrdpath = "stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
#将停用词表转换为list  
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()


# In[48]:


with open('./nlp_test1.txt') as f3:
    res1 = f3.read()
print res1


# In[49]:


with open('./nlp_test2.txt') as f:
    document2 = f.read()
    
    document2_decode = document2.decode('GBK')
    document2_cut = jieba.cut(document2_decode)
    #print  ' '.join(jieba_cut)
    result = ' '.join(document2_cut)
    result = result.encode('utf-8')
    with open('./nlp_test3.txt', 'w') as f2:
        f2.write(result)
f.close()
f2.close()  


# In[50]:


with open('./nlp_test3.txt') as f4:
    res2 = f4.read()
print res2


# In[51]:


jieba.suggest_freq('桓温', True)
with open('./nlp_test4.txt') as f:
    document3 = f.read()
    
    document3_decode = document3.decode('GBK')
    document3_cut = jieba.cut(document3_decode)
    #print  ' '.join(jieba_cut)
    result = ' '.join(document3_cut)
    result = result.encode('utf-8')
    with open('./nlp_test5.txt', 'w') as f3:
        f3.write(result)
f.close()
f3.close()  


# In[52]:


with open('./nlp_test5.txt') as f5:
    res3 = f5.read()
print res3


# In[53]:


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [res1,res2]
vector = TfidfVectorizer(stop_words=stpwrdlst)
tfidf = vector.fit_transform(corpus)
print tfidf


# In[54]:


wordlist = vector.get_feature_names()#获取词袋模型中的所有词  
# tf-idf矩阵 元素a[i][j]表示j词在i类文本中的tf-idf权重
weightlist = tfidf.toarray()  
#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
for i in range(len(weightlist)):  
    print "-------第",i,"段文本的词语tf-idf权重------"  
    for j in range(len(wordlist)):  
        print wordlist[j],weightlist[i][j]  


# In[55]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
corpus = [res1,res2,res3]
cntVector = CountVectorizer(stop_words=stpwrdlst)
cntTf = cntVector.fit_transform(corpus)
print cntTf


# In[61]:


lda = LatentDirichletAllocation(n_topics=2, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
docres = lda.fit_transform(cntTf)


# In[62]:


print lda.components_


# In[63]:


print docres

