# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + hide_input=true
# Essentials
import gc
import numpy as np
import pandas as pd
import datetime
import random
import warnings
import string
from skopt.space import Real, Categorical, Integer
warnings.filterwarnings("ignore")
import functools
import dask
import os
CORE_NUM = int(os.environ['NUMBER_OF_PROCESSORS'])

# Plots
import seaborn as sns
import matplotlib.pyplot as plt

# Models
from sklearn.ensemble import RandomForestRegressor

# Stats
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Tools and metrics
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error
from sklearn.preprocessing import PowerTransformer
from skopt import BayesSearchCV

#bbq packages
import helper as hp
import stats
import product_info as pi
# + hide_input=true
# Purge 交叉验证函数
# TODO: make GitHub GIST
# TODO: add as dataset
# TODO: add logging with verbose

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]
                
                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size
 
            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]
            
            
            if self.verbose > 0:
                    pass
                    
            yield [int(i) for i in train_array], [int(i) for i in test_array]

# + hide_input=true
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
    
# this is code slightly modified from the sklearn docs here:
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#sphx-glr-auto-examples-model-selection-plot-cv-indices-py
def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """Create a sample plot for indices of a cross-validation object."""
    
    cmap_cv = plt.cm.coolwarm

    jet = plt.cm.get_cmap('jet', 256)
    seq = np.linspace(0, 1, 256)
    _ = np.random.shuffle(seq)   # inplace
    cmap_data = ListedColormap(jet(seq))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=plt.cm.Set3)

    ax.scatter(range(len(X)), [ii + 2.5] * len(X),
               c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['target', 'day']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2], xlim=[0, len(y)])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


# -

# # 1 数据处理

# ## 1.1读取数据

# + hide_input=true
data = pd.read_csv("ag2012.csv",index_col='Unnamed: 0')
# -


# 数据有74684行，125列

# + hide_input=true
data.shape

# + hide_input=true
data.tail()

# + hide_input=true
data.head()
# -

# # 2 探索性数据分析

# ## 2.1查看特征

data.iloc[100:105,0:10]

data.iloc[100:105,10:20]

data.iloc[100:105,20:30]

data.iloc[100:105,30:40]

data.iloc[100:105,40:50]

data.iloc[100:105,50:60]

data.iloc[100:105,60:70]

data.iloc[100:105,70:80]

data.iloc[100:105,70:80]

data.iloc[100:105,80:90]

data.iloc[100:105,90:100]

data.iloc[100:105,100:110]

data.iloc[100:105,110:120]

data.iloc[100:105,120:126]

# ## 2.2 查看空值

# + hide_input=true
missing=data.isnull().sum()
missing[data.isnull().sum()!=0].sort_values()
# -

# 前88分钟有很多预热特征值为0 我们从第100个样本开始划分训练集和测试集

# ## 2.3 偏度峰度

# + hide_input=true
data.iloc[:,10:62].skew()
kurtosis = data.iloc[:,10:62].kurtosis().sort_values().to_frame(name="kur")
features = kurtosis.query(' 0 < kur < 10')
features
# -

# 我们获取Kurtosis在0到10直接的特征。如果Kurtosis太小，则特征过于简单，如果Kurtosis太大，则会导致交易频繁

# ## 2.4 弱平稳性检验

import statsmodels.tsa.stattools as ts
import math

features = features.drop(['CorrPV','CorrPOI','CorrAutoRtn'], axis=0)

for i in range(len(features)):
    #result[i] = ts.adfuller(data.iloc[:,i], maxlag=int(pow(len(data)-1,(1/3))), regression='ct', autolag=None)
    print(features.index[i], "adf",ts.adfuller(data.loc[100:,features.index[i]], maxlag=int(pow(len(data)-1,(1/3))), regression='ct', autolag=None))
    #ts.kpss(data.iloc[:,10], regression='c', lags=int(3*math.sqrt(len(data))/13))
    print(features.index[i], "kpss", ts.kpss(data.loc[100:,features.index[i]], regression='c', lags=int(3*math.sqrt(len(data))/13)))

# 有些特征KPSS显著，但是看了ADF是平稳的。

# ## 2.5 对数收益率

# + hide_input=true
ret = (np.log(data['ClosePrice']) - np.log(data['OpenPrice'])).reset_index(drop = True)
ret
# -

# 非零对数收益率

# + hide_input=true
ret_nonZero = ret.to_frame()
ret_nonZero['ret'] = ret_nonZero.iloc[:,0]
ret_nonZero = ret_nonZero.query('ret != 0')#.reset_index(drop = True)
ret_nonZero['ret'] 
# -

# ## 2.6 成交均价走势

# + hide_input=true
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot((pd.Series(data['ClosePrice'])), lw=3, color='red')
ax.set_title ("Ag", fontsize=22);
ax.set_ylabel ("Close Price per min", fontsize=18);

gc.collect();
# -

# ## 2.7 每分钟对数收益率

# + hide_input=true
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot((pd.Series(ret)), lw=1, color='green')
ax.axhline(y=0, linestyle='--', alpha=0.3, c='green', lw=2)
ax.set_title ("Ag", fontsize=22);
ax.set_ylabel ("Avg Log Return", fontsize=18);

gc.collect();
# -

# ## 2.8 每分钟涨跌概率

# + hide_input=true
print("Ag每分钟涨的概率",sum(ret>0)/len(ret))

# + hide_input=true
print('Ag 每分钟收益率为0的情况',sum(ret == 0)/len(ret))

# + hide_input=true
plt.figure(figsize = (12,5))
ax = sns.distplot(ret , 
             bins=3000, 
             kde_kws={"clip":(-0.03,0.03)}, 
             hist_kws={"range":(-0.03,0.03)},
             color='darkcyan', 
             kde=False);
values = np.array([rec.get_height() for rec in ax.patches])
norm = plt.Normalize(values.min(), values.max())
colors = plt.cm.jet(norm(values))
for rec, col in zip(ax.patches, colors):
    rec.set_color(col)
plt.xlabel("Histogram of the resp values", size=14)
plt.show();
#del values
gc.collect();
# -

# 收益率17%情况为0导致收益率分布非常尖峰态

# + hide_input=true
min_resp = ret.min()
print('The minimum value for ret is: %.5f' % min_resp)
max_resp = ret.max()
print('The maximum value for ret is:  %.5f' % max_resp)

# + hide_input=true
print("Skew of resp is:      %.2f" %ret.skew() )
print("Kurtosis of resp is: %.2f"  %ret.kurtosis() )
# -

# 删除收益率为0的收益率分布，依然是尖峰态，有许多收益靠近0，

# + hide_input=true
plt.figure(figsize = (12,5))
ax = sns.distplot(ret_nonZero['ret'] , 
             bins=3000, 
             kde_kws={"clip":(-0.03,0.03)}, 
             hist_kws={"range":(-0.03,0.03)},
             color='darkcyan', 
             kde=False);
values = np.array([rec.get_height() for rec in ax.patches])
norm = plt.Normalize(values.min(), values.max())
colors = plt.cm.jet(norm(values))
for rec, col in zip(ax.patches, colors):
    rec.set_color(col)
plt.xlabel("Histogram of the resp(non-zero) values", size=14)
plt.show();
#del values
gc.collect();

# + hide_input=true
print("Skew of resp(non-zero) is:      %.2f" %ret_nonZero['ret'].skew() )
print("Kurtosis of resp(non-zero) is: %.2f"  %ret_nonZero['ret'].kurtosis() )
# -

# ## 2.9 特征24：NetBuyVolumn特征93%是0

# + hide_input=true
data['NetBuyVolume'].value_counts()

# + hide_input=true
((data[data['NetBuyVolume'] ==0].count()[0])/len(data)).round(2)
# -

# 我们考虑直接删除这个特征

# # 3 特征工程

# 目前所通过统计检验的特征列表

# + hide_input=true
features_index = []
for i in range(len(features)):
    features_index.append(features.index[i])
features_index
# -

# ## 3.1Drop特征

# + hide_input=true
#drop return
data['ret'] = (np.log(data['ClosePrice']) - np.log(data['OpenPrice']))
ret = data['ret']

# + hide_input=true
new_data = data.loc[100:,features_index].reset_index(drop=True)
new_data
# -

# 最终我们选取14个特征进行建模

# # 4 建模

# ## 4.1 划分训练集和测试集

# 前80%数据为训练集，后20%数据为测试集

# + hide_input=true
l = 0.8*len(new_data)
#l

# + hide_input=true
y_train = ret.iloc[1:59667].reset_index(drop = True)
y_test = ret.iloc[59767:74584].reset_index(drop = True)

x_train = new_data.iloc[0:59666].reset_index(drop = True)
x_test = new_data.iloc[59766:74583].reset_index(drop = True)
# -

# ## 4.2 随机森林

# + hide_input=true
# %%time
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor()
#regr_data = regr.fit(x_train,y_train)
regr.fit(x_train,y_train)
RandomForestRegressor(random_state=1)

# + hide_input=true
from statlearning import plot_feature_importance
plot_feature_importance(regr, x_train.columns)
plt.show()

# +
# %%time

tuning_parameters = {
     'max_depth': [10,20,40],
     'min_samples_leaf': [2,4],
     'min_samples_split': [2,5],
     'n_estimators':  [200,400,600,800]
      }

rf_search = RandomizedSearchCV(regr, tuning_parameters, n_iter = 32, cv = 2, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_search.fit(x_train, y_train)

rf = rf_search.best_estimator_

print('Best parameters found by randomised search:', rf_search.best_params_, '\n')

# -

# ## 4.3Optuna Hyperparam Search for XGBoost

# 我们使用 PurgedGroupTimeSeriesSplit 进行时序交叉验证 并用Optuna贝叶斯超参数搜索

# + hide_input=true
import xgboost as xgb
import optuna
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
#groups = x_train['Date'].values
groups = x_train.index.values
# -

cv = PurgedGroupTimeSeriesSplit(
    n_splits=5,
    max_train_group_size=18000,
    group_gap=2000,
    max_test_group_size=6000
)

from sklearn.preprocessing import StandardScaler
def objective(trial, cv=cv, cv_fold_func=np.average):

    # Optuna suggest params
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 350, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.10),
        'subsample': trial.suggest_uniform('subsample', 0.50, 0.90),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.50, 0.90),
        #'gamma': trial.suggest_int('gamma', 0, 20),
        #'missing': -999,
        #'tree_method': 'gpu_hist'  
    }
    
    # setup the pieline
    scaler = StandardScaler()
    clf = xgb.XGBRegressor(**params)

    pipe = Pipeline(steps=[
        ('scaler', scaler),
        ('xgb', clf)
    ])


    # fit for all folds and return composite RMSE score
    rmses = []
    for i, (train_idx, valid_idx) in enumerate(cv.split(
        x_train,
        y_train,
        groups=groups)):
        
        train_data = x_train.iloc[train_idx, :], y_train.iloc[train_idx]
        valid_data = x_train.iloc[valid_idx, :], y_train.iloc[valid_idx]
        
        _ = pipe.fit(x_train.iloc[train_idx, :], y_train.iloc[train_idx])
        preds = pipe.predict(x_train.iloc[valid_idx, :])
        
        rmse = np.sqrt(mean_squared_error(y_train.iloc[valid_idx], preds))
        rmses.append(rmse)
    
    print(f'Trial done: RMSE values on folds: {rmses}')
    return cv_fold_func(rmses)

gc.collect()

np.seterr(over='ignore')

# +
# %%time

FIT_XGB = True

n_trials = 10

if FIT_XGB:
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print(" {}: {}".format(key, value))

# +
best_params = trial.params

#best_params['missing'] = -999
#best_params['tree_method'] = 'gpu_hist' 
# -

best_params

# + hide_input=true
fig, ax = plt.subplots()

cv = PurgedGroupTimeSeriesSplit(
    n_splits=5,
    max_train_group_size=18000,
    group_gap=2000,
    max_test_group_size=6000
)

plot_cv_indices(
    cv,
    x_train,y_train,
    x_train.index.values,
    ax,
    5,
    lw=20
);
# +
##Fit the XGBoost Classifier with Optimal Hyperparams
scaler = StandardScaler()

clf = xgb.XGBRegressor(**best_params)

pipe_xgb = Pipeline(steps=[
    ('scaler', scaler),
    ('xgb', clf)
])

pipe_xgb.fit(x_train,y_train)

gc.collect()
# -

# ## 4.4 XGboost Random search

# %%time
xbst = xgb.XGBRegressor(reg_lambda=0,objective='reg:squarederror', random_state=1)
xbst.fit(x_train,y_train)

# ## 4.5 模型评估

# + hide_input=false
from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols.fit(x_train, y_train)

from sklearn.metrics import mean_squared_error, r2_score,  mean_absolute_error

# Initialise table
columns=['Train RMSE*10^3', 'Test RMSE*10^3','Train MAE*10^3','Test MAE*10^3','Train R^2','Test R^2']
rows=['OLS', 'Random Forest', 'XGBoost(Bayes)', 'XGBoost']
results =pd.DataFrame(0.0, columns=columns, index=rows)

# List algorithms
methods = [ols, rf, pipe_xgb, xbst] 

# Computer test predictions and metrics
for i, method in enumerate(methods):
    y_pred_train = method.predict(x_train)
    y_pred_test = method.predict(x_test)
    
    results.iloc[i, 0] = np.sqrt(mean_squared_error(y_train,y_pred_train))*1000
    results.iloc[i, 1] = np.sqrt(mean_squared_error(y_test,y_pred_test))*1000
    results.iloc[i, 2] = mean_absolute_error(y_train, y_pred_train)*1000
    results.iloc[i, 3] = mean_absolute_error(y_test, y_pred_test)*1000
    results.iloc[i, 4] = r2_score(y_train, y_pred_train)
    results.iloc[i, 5] = r2_score(y_test, y_pred_test) 
    
results.round(6)
# -

# # 5 固定金额回测

# ## 5.1模型预测

# + hide_input=false
#OLS
predict_ols = pd.Series(ols.predict(x_test))
#Random Forest
predict_rf = pd.Series(rf.predict(x_test))
# Optuna+Purged XGbosst 
pipe_prod = pipe_xgb
predict_xgbt = pd.Series(pipe_prod.predict(x_test))
# Random search XGboost
predict_xg = pd.Series(xbst.predict(x_test))
# Neural Network
# -

# ## 5.2 回测

# + hide_input=false
open_price = data['OpenPrice']
open_price_test = open_price.iloc[59767:74584].reset_index(drop=True)
#open_price_test
# -

from collections import OrderedDict
def get_daily_pnl(testx,testy, period=5, tranct_ratio=False,threshold=0.001, tranct=1.1e-4, noise=0, notional=False,invest = 100):
    n_bar = len(testx)
    price = open_price_test#pd.Series(testx['ClosePrice'].astype('int64')).reset_index(drop=True)
    
    #过去5分钟收益率（滚动）
    ret_5 = (testy.rolling(period).sum()).dropna().reset_index(drop=True)
    ret_5 = ret_5.append(pd.Series([0]*(len(testy)-len(ret_5)))).reset_index(drop=True) 
    #ret_5 = testy
    
    #交易信号 过去5分钟收益大于阈值买入，过去5分钟收益小于负阈值卖出
    signal = pd.Series([0] * n_bar)
    signal[(ret_5>threshold)] = 1
    signal[(ret_5< -threshold)] = -1
   
    #买仓
    position_pos = pd.Series([np.nan] * n_bar)
    position_pos[0] = 0 
    position_pos[(signal==1)] = 1
    position_pos[(ret_5< -threshold)] = 0
    position_pos.ffill(inplace=True)
    
    pre_pos = position_pos.shift(1)#前一分钟持仓情况
    position_pos[(position_pos==1) & (pre_pos==1)] = np.nan #如果前一分钟持有，并且交易信号是1，不执行交易
    position_pos[(position_pos==1)] = invest/price[(position_pos==1)]
    position_pos.ffill(inplace=True)
        
    #卖仓
    position_neg = pd.Series([np.nan] * n_bar)
    position_neg[0] = 0
    position_neg[(signal==-1)] = -1
    position_neg[(ret_5> threshold)] = 0
    position_neg.ffill(inplace=True)
    
    pre_neg = position_neg.shift(1)
    position_neg[(position_neg==-1) & (pre_neg==-1)] = np.nan
    position_neg[(position_neg==-1)] = -invest/price[(position_neg==-1)]
    position_neg.ffill(inplace=True)
    
    #持仓
    position = position_pos + position_neg
    position[0]=0
    position[n_bar-1] = 0 #交易结束前平仓
    position[n_bar-2] = 0
    change_pos = position - position.shift(1)
    change_pos[0] = 0
    change_base = pd.Series([0] * n_bar)
    change_buy = change_pos>0
    change_sell = change_pos<0

    if (tranct_ratio):
        change_base[change_buy] = price[change_buy]*(1+tranct)
        change_base[change_sell] = price[change_sell]*(1-tranct)
    else:
        change_base[change_buy] = price[change_buy]+tranct
        change_base[change_sell] = price[change_sell]-tranct
    
    final_pnl = -sum(change_base*change_pos)
    pln_invest = final_pnl/invest
    turnover = sum(change_base*abs(change_pos))
    num = sum((position!=0) & (change_pos!=0))
    hld_period = sum(position!=0)
  
    ## finally we combine the statistics into a data frame
    #result = pd.DataFrame({"final.pnl": final_pnl, "turnover": turnover, "num": num, "hld.period": hld_period}, index=[0])
    #result = {"date": date, "final.pnl": final_pnl, "turnover": turnover, "num": num, "hld.period": hld_period}
    result = OrderedDict([ ("pln/invest", pln_invest),("final.pnl", final_pnl), ("turnover", turnover), ("num", num), ("hld.period", hld_period)])
    return result


# ## 5.3 回测结果

# + hide_input=false
# Initialise table
columns=['pln/invest', 'final.pnl', 'turnover','num','hld.period']
rows=['OLS', 'Random Forest', 'XGBoost(Bayes)', 'XGBoost']
results =pd.DataFrame(0.0, columns=columns, index=rows)

# List algorithms
preds = [predict_ols, predict_rf, predict_xgbt, predict_xg] 
# Compute test predictions and metrics
for i in range(len(preds)):
    results.loc[rows[i]] = pd.DataFrame(get_daily_pnl(x_test,testy = preds[i], period=5, tranct_ratio=True, 
                                               threshold= 0.0005, tranct=0.00015, notional=True, invest = 100),index=[rows[i]]).iloc[0,:]
    
results
# -

