
# coding: utf-8

#  Copyright (C)                                                       
#  2016 - 2019 Pinard Liu(liujianping-ok@163.com) 
#  
#  https://www.cnblogs.com/pinard
#  
#  Permission given to modify the code as long as you keep this declaration at the top                               
# 
# 用scikit-learn和pandas学习Ridge回归
# https://www.cnblogs.com/pinard/p/6023000.html

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


print np.arange(1, 11)
print np.arange(0, 10)[:, np.newaxis]   


# In[4]:


print np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis]


# In[11]:


# X is a 10x10 matrix
X = 1. / (np.arange(1, 11) + np.arange(0, 10)[:, np.newaxis])
# y is a 10 x 1 vector
y = np.ones(10)


# In[13]:


n_alphas = 200
# alphas count is 200, 都在10的-10次方和10的-2次方之间
alphas = np.logspace(-10, -2, n_alphas)
print alphas


# In[14]:


clf = linear_model.Ridge(fit_intercept=False)
coefs = []
# 循环200次
for a in alphas:
    #设置本次循环的超参数
    clf.set_params(alpha=a)
    #针对每个alpha做ridge回归
    clf.fit(X, y)
    # 把每一个超参数alpha对应的theta存下来
    coefs.append(clf.coef_)


# In[18]:


ax = plt.gca()

ax.plot(alphas, coefs)
#将alpha的值取对数便于画图
ax.set_xscale('log')
#翻转x轴的大小方向，让alpha从大到小显示
ax.set_xlim(ax.get_xlim()[::-1]) 
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.axis('tight')
plt.show()

