
# coding: utf-8

# #### 本段代码处理广告操作数据

# In[9]:


import pandas as pd
import numpy as np
import os
import gc
from utiles import memoryOptimize,parseTime
from datetime import timedelta
import copy
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot")
path = "../Data/round1/"


# In[18]:


###广告静态数据
def genAdFeature():
    ad_feature = pd.read_csv(path+"ods/ad_static_feature.out",sep="\t",header=None)
    columns = ["aid","create_time","ad_account_id","item_id","category","industry_id","size"]
    ad_feature.columns = columns
    ad_feature["create_time"] = pd.to_datetime((ad_feature["create_time"] + 3600*8),unit="s") #转变时间格式
    ad_feature["date"] = ad_feature["create_time"].dt.date
    return ad_feature


# In[19]:


def genAdOps():
    ####广告操作数据
    adop = pd.read_csv(path+"ods/ad_operation.dat",sep="\t",header=None) 
    columns = ['aid', 'create_time', 'op_type', 'op_columns', 'op_value']
    adop.columns = columns
    return adop


# In[20]:


def createTime(create_time):
    """
    广告操作数据:将时间转变时间格式
    """
    create_time = str(create_time)
    if create_time !="0":
        date = create_time[:4]+"-"+create_time[4:6]+"-"+create_time[6:8]
        timepoint = create_time[8:10]+":"+create_time[10:12]+":"+create_time[12:]
        return date+" "+timepoint
    else:
        return create_time


def cleanAdops(adop,ad_feature):
    """
    初步清洗广告数据，替换创建时间
    """
    print(adop.shape)
    adop["create_time"] = adop["create_time"].map(createTime)
    adop["date"] = adop["create_time"].map(lambda x:'0' if str(x)=='0' else str(x)[:10])
    adop = adop[~((adop["date"]>"2019-02-28")&(adop["date"]<="2019-02-31"))].sort_values(by=["aid","create_time"]) #过滤2019大于2019-02-28的数据
    adop = adop.reset_index(drop=True)
    print(adop.shape)
    
    #关联广告静态数据
    origin_time = ad_feature[["aid","create_time","date"]].rename(columns={"create_time":"og_time","date":"og_date"})
    adop = adop.merge(origin_time,on="aid",how="inner") #关联广告静态数据
    adop.loc[adop["create_time"]=="0","create_time"] = adop.loc[adop["create_time"]=="0","og_time"].map(str) #将创建时间为0的替换成静态广告文件的创建时间
    adop.loc[adop["date"]=="0","date"] = adop.loc[adop["date"]=="0","og_date"].map(str) #替换日期
    adop = adop.drop(["og_time","og_date"],axis=1)
    adop = adop[~((adop["op_columns"]==1)&(adop["op_value"]=="0"))] #过滤使广告失效的记录
    adop = adop.drop_duplicates()
    print(adop.shape)
    return adop


# In[4]:


def getAdSettingValue(adop):
    """
    获取广告各项设定值
    """
    #取每个广告每天各个设置字段最后一个取值
    new_adop = adop.pivot_table(index=["aid","date"],columns=["op_columns"],values="op_value",aggfunc=lambda x:x.values[-1])
    new_adop = new_adop.reset_index()
    new_adop.index= [i for i in range(new_adop.shape[0])]
    new_adop = new_adop.rename(columns={2:"bid",3:"group_person",4:"explosure_time"}) #重命名
    new_adop = new_adop.drop([1],axis=1).drop_duplicates()
    
    #人群设置，如果该字段为空就取最近一次不为空的值代替
    for i in ["bid","group_person","explosure_time"]:
        print(i)
        person_null = new_adop[new_adop[i].isnull()] 
        person_notnull = new_adop[new_adop[i].notnull()]
        person_null = person_null.merge(person_notnull[["aid","date",i]],on="aid",how="left").query("date_x > date_y").drop_duplicates()
        filter_date = person_null.groupby(["aid","date_x"],as_index=False)["date_y"].max() #去离每一条空值记录时间最近的非空值记录
        person_null = person_null.merge(filter_date,on=["aid","date_x","date_y"],how="inner").drop_duplicates() #
        person_null[i + "_x"] = person_null[i+"_y"]
        person_null = person_null.drop(["date_y",i+"_y"],axis=1)
        person_null.columns = person_notnull.columns
        new_adop = pd.concat([person_null,person_notnull],axis=0,ignore_index=True)
        print(new_adop.shape)
        
    new_adop = new_adop.sort_values(by = ["aid","date"])
    return new_adop


# In[40]:


def genDict(column):
    """
    处理人群定向特征
    """
    dict_ = {}
    column_ls = column.split("|")
    for i in column_ls:
        key = i.split(":")[0]
        val = i.split(":")[1]
        dict_[key] = val
    return dict_

def splitPersonFeature(new_adop,dtype="train"):
    df = copy.deepcopy(new_adop)
    df_part1 = df[df["group_person"].str.contains("all")]
    df_part2 = df[~df["group_person"].str.contains("all")]
    for i in ["age","gender","area","status","education","consuptionAbility","device","work","ConnectionType","behavior"]:
        df_part1[i] = "all"

    df_part2["dict_group_person"] = df_part2["group_person"].map(genDict)
    for i in ["age","gender","area","status","education","consuptionAbility","device","work","ConnectionType","behavior"]:
        df_part2[i] = df_part2["dict_group_person"].map(lambda x:x[i] if i in x.keys() else np.nan)

    df = pd.concat([df_part1,df_part2],axis=0,ignore_index=True).reset_index(drop=True)
    if dtype=="train":
        columns1 = ['aid', 'date', 'bid', 'explosure_time']
        columns2 = ["age","gender","area","status","education","consuptionAbility","device","work","ConnectionType","behavior"]
        columns1.extend(columns2)
        df =df[columns1]
        df = df.sort_values(by=["aid","date"])
    else:
        columns1 = ['row_id', 'aid', 'create_time', 'size', 'industry_id', 'category',
                   'item_id', 'ad_ccount_id', 'explosure_time', 'bid']
        columns2 = ["age","gender","area","status","education","consuptionAbility","device","work","ConnectionType","behavior"]
        columns1.extend(columns2)
        df =df[columns1]
        df = df.sort_values(by=["row_id","aid"])
    return df


# In[27]:


def mergeExplosure(new_adop):
    """
    关联爆光日志
    """
    explosure = pd.read_pickle(path+"/clean/explosure_1.pickle")
    tmp = explosure[["aid","timestamp"]]
    del explosure
    tmp["date"] = tmp["timestamp"].dt.date
    tmp_aid_cnt = tmp.groupby(["aid","date"],as_index=False)["timestamp"].count().rename(columns={"timestamp":"imp_cnt","date":"next_date"}) #统计每个广告每天的爆光量
    tmp_aid_cnt["next_date"] = tmp_aid_cnt["next_date"].map(str)
    
    new_adop["next_date"] = pd.to_datetime(new_adop["date"]) + timedelta(days=1)
    new_adop["next_date"] = new_adop["next_date"].map(lambda x:str(x).split(" ")[0])
    setting_feature = new_adop.merge(tmp_aid_cnt,left_on=["aid","next_date"],right_on=["aid","next_date"],how="inner")
    return setting_feature


# In[21]:


adop = genAdOps() #获取广告操作数据
ad_feature = genAdFeature() #获取广告静态数据
adop = cleanAdops(adop,ad_feature) #初步清洗广告
new_adop = getAdSettingValue(adop) #获取广告设置内容
new_adop = splitPersonFeature(new_adop)  #将人群定向内容进行分解
new_adop = mergeExplosure(new_adop) #关联爆光日志


# In[31]:


# new_adop.to_pickle("../Data/round1/clean/setting_feature_0.pickle")


# #### 测试集

# In[32]:


def genTest():
    test_example = pd.read_csv(path+"ods/test_sample.dat",sep="\t",header=None)
    columns = ["row_id","aid","create_time","size","industry_id",
           "category","item_id","ad_ccount_id","explosure_time","group_person","bid"]
    test_example.columns = columns
    return test_example


# In[33]:


test = genTest()


# In[42]:


new_test = splitPersonFeature(test,"test")


# In[44]:


new_test.to_pickle("../Data/round1/clean/test_setting_feature_0.pickle")

