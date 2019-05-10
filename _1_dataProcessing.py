
# coding: utf-8

# #### 本notebook主要对原始的数据进行简单的处理

# In[1]:


import pandas as pd
import numpy as np
import os
import gc
from utiles import memoryOptimize,parseTime
import copy
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[42]:


path = "../Data/round1/"


# ### 1.读取用户表

# In[5]:


get_ipython().run_cell_magic('time', '', 'user = pd.read_csv(path+"/ods/user_data",sep="\\t",encoding="utf-8",header=None)\ncolumns = ["uid","age","gender","area","status","education",\n           "consuptionAbility","device","work","connectionType","behavior"]\nuser.columns = columns\nuser.head()')


# #### 1.1 缩减内存

# In[6]:


get_ipython().run_cell_magic('time', '', 'user = memoryOptimize(user)')


# In[10]:


# user.to_pickle(path+"clean/user.pickle")


# In[4]:


get_ipython().run_cell_magic('time', '', 'user = pd.read_pickle(path+"clean/user.pickle")')


# In[7]:


user.head()


# In[8]:


user.nunique()


# #### 1.2 对area进行分析

# In[29]:


get_ipython().run_cell_magic('time', '', 'tmp = user["area"].map(lambda x:str(x).split(",")) #将area进行分割\ntmp_len = user["area"].map(lambda x:len(str(x).split(","))) #将area分割后的标签进行统计\nprint("area组合个数最多为:",max(tmp_len))\nprint("area组合个数最少为:",min(tmp_len))\n#获取area\ntmp = tmp.tolist()\narea_ls = []\nfor i in tmp:\n    area_ls.extend(i)')


# ##### area切割后有13646不同的取值

# In[30]:


area_set = set(area_ls)
print(len(area_set))


# ##### 计算每个area的词频

# In[31]:


area_series = pd.Series(area_ls)
area_series.value_counts()


# ### 2.爆光日志

# In[ ]:


get_ipython().run_cell_magic('time', '', 'explosure = pd.read_pickle(path+"/clean/explosure_1.pickle")\ncolumns = ["resid","time","posid","uid","aid","size","bid","pctr","quality_ecpm"," total_ecpm"]\nexplosure.columns = columns')


# In[ ]:


explosure = memory_optimze(explosure)
explosure["timestamp"] = pd.to_datetime(explosure["time"] + 3600*8,unit="s")
del explosure["time"]
# explosure.to_pickle(path+"explosure.pickle")


# In[5]:


get_ipython().run_cell_magic('time', '', 'explosure = pd.read_pickle(path+"/clean/explosure_1.pickle")')


# In[6]:


explosure.head()


# In[7]:


tmp = explosure[["timestamp"]]
tmp = parseTime(tmp)


# #### 统计每天总的曝光量

# In[14]:


# date_cnt = tmp["date"].value_counts().sort_index()
date_cnt.plot()


# In[ ]:


date_cnt = tmp["date"].value_counts().sort_index()


# ##### 统计各个小时分布

# In[16]:


tmp[tmp["date"]=="2019-02-28"]["hour"].value_counts().sort_index().plot()


# #### 3修改广告记录

# In[53]:


adop = pd.read_csv(path+"ods/ad_operation.dat",sep="\t",header=None) 
columns = ['aid', 'create_time', 'op_type', 'op_columns', 'op_value']
adop.columns = columns


# In[55]:


def createTime(create_time):
    create_time = str(create_time)
    if create_time !="0":
        date = create_time[:4]+"-"+create_time[4:6]+"-"+create_time[6:8]
        timepoint = create_time[8:10]+":"+create_time[10:12]+":"+create_time[12:]
        return date+" "+timepoint
    else:
        return create_time


# In[56]:


adop["create_time"] = adop["create_time"].map(createTime)


# In[62]:


adop["hour"] = adop["create_time"].map(lambda x:'-1' if x=='0' else x.split(" ")[1].split(":")[0])


# In[64]:


adop["hour"].value_counts().sort_index().plot("bar")


# In[71]:


adop[adop["hour"]=='05']["op_columns"].value_counts()


# In[74]:


adop[(adop["hour"]=='05')&(adop["op_columns"]==1)]["op_value"].value_counts()


# In[77]:


adop[(adop["hour"]=='05')&(adop["op_columns"]==1)]


# In[75]:


adop[(adop["hour"]=='00')&(adop["op_columns"]==1)]["op_value"].value_counts()


# In[76]:


adop[(adop["hour"]=='23')&(adop["op_columns"]==1)]["op_value"].value_counts()


# #### op_type=2的时候为创建新广告，create_time为0

# In[67]:


adop[adop["op_type"]==2]["create_time"].value_counts()


# #### 广告静态属性

# In[28]:


ad_feature = pd.read_csv(path+"ods/ad_static_feature.out",sep="\t",header=None)
columns = ["aid","create_time","ad_account_id","item_id","category","industry_id","size"]
ad_feature.columns = columns
ad_feature["create_time"] = pd.to_datetime((ad_feature["create_time"] + 3600*8),unit="s")
ad_feature.head()


# #### test_example

# In[35]:


test_example = pd.read_csv(path+"ods/test_sample.dat",sep="\t",header=None)
columns = ["row_id","aid","create_time","size","industry_id",
           "category","item_id","ad_ccount_id","explosure_time","group_person","price_score"]
test_example.columns = columns


# In[38]:


test_example.head()


# In[40]:


test_example.to_pickle(path+"clean/test_exmple.pickle")


# In[80]:


test_example["aid"].value_counts()


# In[81]:


test_example[test_example["aid"]==540848]


# In[12]:


industry_i

