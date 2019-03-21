
# coding: utf-8

# Copyright (C) 2016 - 2019 Pinard Liu(liujianping-ok@163.com)
# 
# https://www.cnblogs.com/pinard
# 
# Permission given to modify the code as long as you keep this declaration at the top
# 
# 用scikit-learn学习K-Means聚类 https://www.cnblogs.com/pinard/p/6169370.html

# In[47]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.datasets.samples_generator import make_blobs
# X为样本特征，Y为样本簇类别， 共1000个样本，每个样本4个特征，共4个簇，簇中心在[-1,-1], [0,0],[1,1], [2,2]， 簇方差分别为[0.4, 0.2, 0.2]
X, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1,-1], [0,0], [1,1], [2,2]], cluster_std=[0.4, 0.2, 0.2, 0.2], 
                  random_state =9)
plt.scatter(X[:, 0], X[:, 1], marker='o')
plt.show()


# In[31]:


from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=2, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()


# In[32]:


from sklearn import metrics
metrics.calinski_harabaz_score(X, y_pred)  


# In[33]:


from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=3, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()


# In[34]:


metrics.calinski_harabaz_score(X, y_pred)  


# In[35]:


from sklearn.cluster import KMeans
y_pred = KMeans(n_clusters=4, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()


# In[36]:


metrics.calinski_harabaz_score(X, y_pred)  


# In[37]:


from sklearn.cluster import MiniBatchKMeans
y_pred = MiniBatchKMeans(n_clusters=2, batch_size = 200, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()


# In[38]:


metrics.calinski_harabaz_score(X, y_pred)  


# In[39]:


from sklearn.cluster import MiniBatchKMeans
y_pred = MiniBatchKMeans(n_clusters=3, batch_size = 200, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()


# In[40]:


metrics.calinski_harabaz_score(X, y_pred)  


# In[41]:


from sklearn.cluster import MiniBatchKMeans
y_pred = MiniBatchKMeans(n_clusters=4, batch_size = 200, random_state=9).fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.show()


# In[42]:


metrics.calinski_harabaz_score(X, y_pred)  


# In[75]:


plt.subplots_adjust(left=.02, right=.98, bottom=.096, top=.96, wspace=.05,
                    hspace=.01)
plt.subplot(2,2,1)
y_pred = MiniBatchKMeans(n_clusters=2, batch_size = 200, random_state=9).fit_predict(X)
score2= metrics.calinski_harabaz_score(X, y_pred)  
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('k=%d, score: %.2f' % (2,score2)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')


plt.subplot(2,2,2)
y_pred = MiniBatchKMeans(n_clusters=3, batch_size = 200, random_state=9).fit_predict(X)
score3= metrics.calinski_harabaz_score(X, y_pred)  
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('k=%d, score: %.2f' % (3,score3)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')

plt.subplot(2,2,3)
y_pred = MiniBatchKMeans(n_clusters=4, batch_size = 200, random_state=9).fit_predict(X)
score4= metrics.calinski_harabaz_score(X, y_pred)  
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('k=%d, score: %.2f' % (4,score4)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')

plt.subplot(2,2,4)
y_pred = MiniBatchKMeans(n_clusters=5, batch_size = 200, random_state=9).fit_predict(X)
score5 = metrics.calinski_harabaz_score(X, y_pred)  
plt.scatter(X[:, 0], X[:, 1], c=y_pred)
plt.text(.99, .01, ('k=%d, score: %.2f' % (5,score5)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
plt.show()


# In[87]:


plt.subplots_adjust(left=.02, right=.98, bottom=.096, top=.96, wspace=.1,
                    hspace=.1)
for index, k in enumerate((2,3,4,5)):
    plt.subplot(2,2,index+1)
    y_pred = MiniBatchKMeans(n_clusters=k, batch_size = 200, random_state=9).fit_predict(X)
    score= metrics.calinski_harabaz_score(X, y_pred)  
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.text(.99, .01, ('k=%d, score: %.2f' % (k,score)),
                 transform=plt.gca().transAxes, size=10,
                 horizontalalignment='right')
plt.show()

