
# coding: utf-8

# 
# Copyright (C)
# 2016 - 2019 Pinard Liu(liujianping-ok@163.com)
# 
# https://www.cnblogs.com/pinard
# 
# Permission given to modify the code as long as you keep this declaration at the top
# 
# 矩阵分解在协同过滤推荐算法中的应用 https://www.cnblogs.com/pinard/p/6351319.html

# In[1]:


import os
import sys

#下面这些目录都是你自己机器的Spark安装目录和Java安装目录
os.environ['SPARK_HOME'] = "C:/Tools/spark-2.2.0-bin-hadoop2.6/"
os.environ['PYSPARK_PYTHON'] = "C:/Users/tata/AppData/Local/Programs/Python/Python36/python.exe"
os.environ['HADOOP_HOME'] = "C:/Tools/hadoop-2.6.0"

sys.path.append("C:/Tools/spark-2.2.0-bin-hadoop2.6/bin")
sys.path.append("C:/Tools/spark-2.2.0-bin-hadoop2.6/python")
sys.path.append("C:/Tools/spark-2.2.0-bin-hadoop2.6/python/pyspark")
sys.path.append("C:/Tools/spark-2.2.0-bin-hadoop2.6/python/lib")
sys.path.append("C:/Tools/spark-2.2.0-bin-hadoop2.6/python/lib/pyspark.zip")
sys.path.append("C:/Tools/spark-2.2.0-bin-hadoop2.6/python/lib/py4j-0.10.4-src.zip")
sys.path.append("C:/Program Files/Java/jdk1.8.0_171")

from pyspark import SparkContext
from pyspark import SparkConf


sc = SparkContext("local","testing")


# In[2]:


print (sc)


# In[5]:


user_data = sc.textFile("C:/Temp/ml-100k/u.data")
user_data.first()


# In[6]:


rates = user_data.map(lambda x: x.split("\t")[0:3])
print (rates.first())


# In[7]:


from pyspark.mllib.recommendation import Rating
rates_data = rates.map(lambda x: Rating(int(x[0]),int(x[1]),int(x[2])))
print (rates_data.first())


# In[8]:


from  pyspark.mllib.recommendation import ALS
from pyspark.mllib.recommendation import MatrixFactorizationModel
sc.setCheckpointDir('checkpoint/')
ALS.checkpointInterval = 2
model = ALS.train(ratings=rates_data, rank=20, iterations=5, lambda_=0.02)


# In[9]:


print (model.predict(38,20))


# In[11]:


print (model.recommendProducts(38,10))


# In[12]:


print (model.recommendUsers(20,10))


# In[13]:


print (model.recommendProductsForUsers(3).collect())


# In[15]:


print (model.recommendUsersForProducts(3).collect())

