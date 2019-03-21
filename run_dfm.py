
# coding: utf-8

# In[1]:


#======config=========

SUB_DIR = "./"


NUM_SPLITS = 3
RANDOM_SEED = 233

# types of columns of the dataset dataframe
CATEGORICAL_COLS = [
   'pass_identity', #category
'bdg_age_range',#category
'eight_crowd', #category
'bdg_consumption_level', #category
'h_channels_s',#category
'setup_ele',#category
'setup_meituan',#category
'bdg_professioin',#category 
'poi_1',
'poi_2',
'poi_3',
'poi_4',
'poi_5',
]

NUMERIC_COLS = ['user_health_value',
'privilege_sensitive',
'avg_total_price_a',
'avg_total_price_q',
'avg_total_price_m',
'avg_total_price_s',
'avg_pay_price_a',
'avg_pay_price_q',
'avg_pay_price_m',
'avg_pay_price_s',
'wuyouhui_order_ratio_a',
'wuyouhui_order_ratio_q',
'wuyouhui_order_ratio_m',
'wuyouhui_order_ratio_s',
'order_rate',
'privilege_sensitive.1',
'next_month_avg_ue',
'this_month_avg_ue',
'price_preference_score',
'service_preference_score',
'resource_preference_score',
'total_coupon_price',
'total_coupon_balance',
'total_coupon_used_ratio_a',
'total_coupon_used_ratio_q',
'total_coupon_used_ratio_m',
'total_coupon_used_ratio_s',
'price_preference_score.1',
'visit_frequency_q',
'total_order_q',
'net_worth_score',
'wuliu_order_ratio_a',
'total_shop_butie_ratio_m',
'tizao_total_order_dinner_q',
'shop_concentration_q',
'avg_profit_a',
'sum_shop_butie_m',
'first_day',
'tizao_med_pay_price_dinner_q',
'total_shop_butie_ratio_q',
'sum_shop_butie_s',
'max_delivery_price_m',
'total_shop_butie_ratio_a',
'min_delivery_price_q',
'max_delivery_price_a',
'avg_net_profit_a',
'max_delivery_price_q',
'avg_wuliu_take_out_time_s',
'avg_wuliu_take_out_time_q',
'avg_wuliu_dis_s',
'last_delay_day_a',
'shop_concentration_a',
'loss_ratio_q',
'avg_net_profit_q',
'tizao_avg_pay_price_dinner_q',
'min_delivery_price_a',
'loss_ratio_m',
'tizao_total_order_dinner_s',
'lat_1',
'lat_2',
'lat_3',
'lat_4',
'lat_5',
'lat_6',
'lat_7',
'lat_8',
'lat_9',
'lat_10',
'lng_1',
'lng_2',
'lng_3',
'lng_4',
'lng_5',
'lng_6',
'lng_7',
'lng_8',
'lng_9',
'lng_10',
'doc_0',
'doc_1',
'doc_2',
'doc_3',
'doc_4',
'doc_5',
'doc_6',
'doc_7',
'doc_8',
'doc_9',
'doc_10',
'doc_11',
'doc_12',
'doc_13',
'doc_14',
'doc_15',
'doc_16',
'doc_17',
'doc_18',
'doc_19',
]

IGNORE_COLS = []



# In[2]:


from __future__ import unicode_literals
import pandas as pd
import numpy as np
import json
import sys
from sklearn.model_selection import train_test_split
import xgboost as xgb
import sklearn
import os
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.utils import shuffle


import numpy as np
import pandas as pd
from sklearn.model_selection import *





def gini(actual, pred):
    assert (len(actual) == len(pred))
    #np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
    all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
    #np.arange创建等差数组，其实是生成index:0~n-1
    #pred 列取负数：-1 * all[:, 1]
    all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]  #把pred列的分数由高到低排序
    totalLosses = all[:, 0].sum()
    giniSum = all[:, 0].cumsum().sum() / totalLosses

    giniSum -= (len(actual) + 1) / 2.
    return giniSum / len(actual)

def gini_norm(actual, pred):
    return gini(actual, pred) / gini(actual, actual)


def custom_error(preds, dtrain):
    labels = dtrain.get_label()
    

    return 'gini_norm',gini_norm(labels,preds)


df_train = shuffle(pd.read_pickle("./df_train_scaled.pickle"))
#标准化连续值
df_train[NUMERIC_COLS] = StandardScaler().fit_transform(df_train[NUMERIC_COLS])
train_X,test_X = train_test_split(df_train,test_size = 0.2,random_state = 233)


# In[3]:


#==========data_reader========

class FeatureDictionary(object):
    def __init__(self, trainfile=None, testfile=None,
                 dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"
        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dfTrain
        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)
        else:
            dfTest = self.dfTest
        df = pd.concat([dfTrain, dfTest]) #首尾相连
        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                # map to a single index
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
                tc += len(us)
        self.feat_dim = tc


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        if has_label:
            y = dfi["l3"].values.tolist()
            dfi.drop(["l3"], axis=1, inplace=True)
        else:
            ids = dfi.index.tolist()
            
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.tolist()
        if has_label:
            return Xi, Xv, y
        else:
            return Xi, Xv, ids
        
        


# In[4]:


from matplotlib import pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
import tensorflow as tf

gini_scorer = make_scorer(gini_norm, greater_is_better=True, needs_proba=True)
from sklearn.model_selection import train_test_split

def _load_data():
    
    
    dfTrain = train_X
    #dfTest = 


    cols = [c for c in dfTrain.columns if c not in ["l3"]]

    X_train = dfTrain[cols].values
    y_train = dfTrain["l3"].values
    dfTest = test_X[cols]
    X_test = dfTest[cols].values
    ids_test = dfTest.index.tolist()
    cat_features_indices = [i for i,c in enumerate(cols) if c in CATEGORICAL_COLS]

    return dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices


def _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params):
    fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=NUMERIC_COLS,
                           ignore_cols=IGNORE_COLS)
    data_parser = DataParser(feat_dict=fd)
    Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
    Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)

    dfm_params["feature_size"] = fd.feat_dim
    dfm_params["field_size"] = len(Xi_train[0])

    y_train_meta = np.zeros((dfTrain.shape[0], 1), dtype=float)
    y_test_meta = np.zeros((dfTest.shape[0], 1), dtype=float)
    _get = lambda x, l: [x[i] for i in l]
    gini_results_cv = np.zeros(len(folds), dtype=float)
    gini_results_epoch_train = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)
    gini_results_epoch_valid = np.zeros((len(folds), dfm_params["epoch"]), dtype=float)

    for i, (train_idx, valid_idx) in enumerate(folds):
        #valid_idx = valid_idx[:1024]
        Xi_train_, Xv_train_, y_train_ = _get(Xi_train, train_idx), _get(Xv_train, train_idx), _get(y_train, train_idx)
        Xi_valid_, Xv_valid_, y_valid_ = _get(Xi_train, valid_idx), _get(Xv_train, valid_idx), _get(y_train, valid_idx)

        dfm = DeepFM(**dfm_params)
        dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_)
        #dfm.fit(Xi_train_, Xv_train_, y_train_, Xi_valid_, Xv_valid_, y_valid_,early_stopping=True)
        y_train_meta[valid_idx,0]= dfm.predict(Xi_valid_, Xv_valid_)
        y_test_meta[:,0] += dfm.predict(Xi_test, Xv_test)

        gini_results_cv[i] = gini_norm(y_valid_, y_train_meta[valid_idx])
        gini_results_epoch_train[i] = dfm.train_result
        gini_results_epoch_valid[i] = dfm.valid_result

    y_test_meta /= float(len(folds))

    # save result
    if dfm_params["use_fm"] and dfm_params["use_deep"]:
        clf_str = "DeepFM"
    elif dfm_params["use_fm"]:
        clf_str = "FM"
    elif dfm_params["use_deep"]:
        clf_str = "DNN"
    print("%s: %.5f (%.5f)"%(clf_str, gini_results_cv.mean(), gini_results_cv.std()))
    filename = "%s_Mean%.5f_Std%.5f.csv"%(clf_str, gini_results_cv.mean(), gini_results_cv.std())
    _save_result(ids_test, y_test_meta, filename)

    _plot_fig(gini_results_epoch_train, gini_results_epoch_valid, clf_str)

    return y_train_meta, y_test_meta

def _save_result(ids, y_pred, filename="dfm_res.csv"):
    pd.DataFrame({"id": ids, "target": y_pred.flatten()}).to_csv(
    os.path.join(SUB_DIR, filename), index=False, float_format="%.5f")



def _plot_fig(train_results, valid_results, model_name):
    colors = ["red", "blue", "green"]
    xs = np.arange(1, train_results.shape[1]+1)
    plt.figure()
    legends = []
    for i in range(train_results.shape[0]):
        plt.plot(xs, train_results[i], color=colors[i], linestyle="solid", marker="o")
        plt.plot(xs, valid_results[i], color=colors[i], linestyle="dashed", marker="o")
        legends.append("train-%d"%(i+1))
        legends.append("valid-%d"%(i+1))
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Gini")
    plt.title("%s"%model_name)
    plt.legend(legends)
    plt.savefig("./fig/%s.png"%model_name)
    plt.close()




# In[5]:


from DeepFM import DeepFM
from ipykernel import kernelapp as app


# In[6]:


def accuracy(actual, pred):
    dfm_res= []
    for i in pred:
        if i >0.5:
            dfm_res.append(1)
        else:
            dfm_res.append(0)
    print "pred: %d , actual: %d , total : %d" % (sum(dfm_res),sum(actual),len(pred))
    
    s = sklearn.metrics.accuracy_score(actual,dfm_res)
    return s
def gini_norm(actual, pred):
    dfm_res= []
    for i in pred:
        if i >0.5:
            dfm_res.append(1)
        else:
            dfm_res.append(0)
    print "pred: %d , actual: %d , total : %d" % (sum(dfm_res),sum(actual),len(pred))
    return gini(actual, pred) / gini(actual, actual)


# In[7]:


# load data
dfTrain, dfTest, X_train, y_train, X_test, ids_test, cat_features_indices = _load_data()

# folds
folds = list(StratifiedKFold(n_splits=NUM_SPLITS, shuffle=True,
                             random_state=RANDOM_SEED).split(X_train, y_train))
fd = FeatureDictionary(dfTrain=dfTrain, dfTest=dfTest,
                           numeric_cols=NUMERIC_COLS,
                           ignore_cols=IGNORE_COLS)
data_parser = DataParser(feat_dict=fd)
Xi_train, Xv_train, y_train = data_parser.parse(df=dfTrain, has_label=True)
Xi_test, Xv_test, ids_test = data_parser.parse(df=dfTest)



# In[8]:


np.array(Xi_train).shape,np.array(Xv_train).shape,np.array(Xi_test).shape,np.array(Xv_test).shape


# In[9]:


# ------------------ DeepFM Model ------------------

dfm_params = {
    "use_fm": True,
    "use_deep": True,
    "embedding_size": 8,
    "dropout_fm": [1, 1],
    "deep_layers": [32],   
    "dropout_deep": [0.5, 0.5, 0.5],
    "deep_layers_activation": tf.nn.sigmoid,
    "epoch": 10,
    "batch_size": 1024,
    "learning_rate": 0.004,
    "optimizer_type": "adam",
    "batch_norm": 1,
    "batch_norm_decay": 0.997,
    "l2_reg": 0.01,
    "verbose": True,
    "eval_metric": gini_norm,   
    "random_seed": RANDOM_SEED,
    "use_sample_weights": True,
    "sample_weights_dict":{0:1,1:2},
    
}
y_train_dfm, y_test_dfm = _run_base_model_dfm(dfTrain, dfTest, folds, dfm_params)

# # ------------------ FM Model ------------------
# fm_params = dfm_params.copy()
# fm_params["use_deep"] = False
# y_train_fm, y_test_fm = _run_base_model_dfm(dfTrain, dfTest, folds, fm_params)


# # ------------------ DNN Model ------------------
# dnn_params = dfm_params.copy()
# dnn_params["use_fm"] = False
# y_train_dnn, y_test_dnn = _run_base_model_dfm(dfTrain, dfTest, folds, dnn_params)




