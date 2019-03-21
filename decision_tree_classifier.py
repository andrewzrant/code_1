
# coding: utf-8

# Copyright (C)
# 2016 - 2019 Pinard Liu(liujianping-ok@163.com)
# 
# https://www.cnblogs.com/pinard
# 
# Permission given to modify the code as long as you keep this declaration at the top
# 
# scikit-learn决策树算法类库使用小结 https://www.cnblogs.com/pinard/p/6056319.html

# In[1]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd


# In[2]:


from sklearn.datasets import load_iris
from sklearn import tree


# In[3]:


iris = load_iris()


# In[4]:


clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)


# In[5]:


from IPython.display import Image  
import pydotplus 


# In[6]:


dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  


# In[7]:


graph = pydotplus.graph_from_dot_data(dot_data)  


# In[8]:


Image(graph.create_png())  


# In[9]:


dot_data = tree.export_graphviz(clf, out_file=None)
graph = pydotplus.graph_from_dot_data(dot_data) 
graph.write_pdf("iris.pdf") 

