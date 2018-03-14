#-*- encoding: utf-8 -*-

from sklearn import linear_model
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge#岭回归
from sklearn.linear_model import LinearRegression# 线性回归
from sklearn.linear_model import Lasso# 套索回归
from sklearn.linear_model import ElasticNet# 弹性回归网络
from sklearn.ensemble import RandomForestRegressor#随机森林 比线性回归好点
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor #GDBT集成模型
from sklearn.ensemble import AdaBoostRegressor #提升模型
from sklearn.tree import DecisionTreeRegressor#决策树
from sklearn.neighbors import KNeighborsRegressor#knn近邻模型
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from analysis import importData
from matplotlib import pyplot
from sklearn.model_selection import GridSearchCV#网格评估
'''
本页面时学习调参时的一些记录
'''
X,zzz = importData.deal_data_train(r'train.csv')
Y = X.pop('PM_US_Post')
X_train, X_validation, Y_train,Y_validation = train_test_split(X,Y,train_size=0.7,random_state=7)

num_folds = 10
seed = 7
scoring = 'neg_mean_squared_error'

# models = {}
# models['LR'] = LinearRegression()
# models['LASSO'] = Lasso()
# models['RFR'] = RandomForestRegressor()
# models['R'] = Ridge()
# models['EN'] = ElasticNet()
# models['KNN'] = KNeighborsRegressor()
# models['CART'] = DecisionTreeRegressor()
#
# #评估算法
# results = []
# for key in models:
#     kfold = KFold(n_splits=num_folds,random_state=seed)
#     cv_reult = cross_val_score(models[key],X_train,Y_train,cv=kfold,scoring=scoring)
#     results.append(cv_reult)
#     print('%s: %f (%f)' % (key,cv_reult.mean(),cv_reult.std()))
# #画出箱图
# fig = pyplot.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# pyplot.boxplot(results)
# ax.set_xticklabels(models.keys())
# pyplot.show()

#评估算法-正态化数据（pipeline）
# pipelines = {}
# pipelines['ScalerLR'] = Pipeline([('Scaler',StandardScaler()),('LR',LinearRegression())])
# pipelines['ScalerSASSO'] = Pipeline([('Scaler',StandardScaler()),('SASSO',Lasso())])
# pipelines['ScalerRFR'] = Pipeline([('Scaler',StandardScaler()),('RFR',RandomForestRegressor())])
# pipelines['ScalerR'] = Pipeline([('Scaler',StandardScaler()),('R',Ridge())])
# pipelines['ScalerRN'] = Pipeline([('Scaler',StandardScaler()),('EN',ElasticNet())])
# pipelines['ScalerKNN'] = Pipeline([('Scaler',StandardScaler()),('KNN',KNeighborsRegressor())])
# pipelines['ScalerCART'] = Pipeline([('Scaler',StandardScaler()),('CART',DecisionTreeRegressor())])
# results = []
# for key in pipelines:
#     kfold = KFold(n_splits=num_folds,random_state=seed)
#     cv_reult = cross_val_score(pipelines[key],X_train,Y_train,cv=kfold,scoring=scoring)
#     results.append(cv_reult)
#     print('%s: %f (%f)' % (key, cv_reult.mean(), cv_reult.std()))
#算法调参(先把数据正态化)
# scaler = StandardScaler().fit(X_train)
# rescaledX = scaler.transform(X_train)
# #n_estimators是最大迭代次数
# #param_test1 = {'n_estimators':range(10,71,10)}#最优:-2075.23435594 使用{'n_estimators': 50}
# param_test2 = {'max_depth':range(3,15,1),'min_samples_split':range(50,201,20)}#最优:-3127.97090623 使用{'max_depth': 14, 'min_samples_split': 50}
# model = RandomForestRegressor()
# #用网格调参和k折交叉验证评估
# kford = KFold(n_splits=num_folds,random_state=seed)
# grid = GridSearchCV(estimator=model,param_grid=param_test2,scoring=scoring,cv=kford)
# grid_result = grid.fit(X = rescaledX,y = Y_train)
#
# print('最优:%s 使用%s'%(grid_result.best_score_,grid_result.best_params_))
# cv_result=zip(grid_result.cv_results_['mean_test_score'],grid_result.cv_results_['std_test_score'],grid_result.cv_results_['std_test_score'])
# for mean ,std,param in cv_result:
#     print('%f (%f) with %r  '%(mean,std,param))

# reg =RandomForestRegressor(n_estimators=50)#0.776661778787
# reg.fit(X_train,Y_train)
# pre_y = reg.predict(X_validation)
# print(r2_score(Y_validation,pre_y))
#选用集成算法 分别测集成adaboost和决策树，以及两者在一起的分数。
# ensembles = {}
# ensembles['ScaledAB']= Pipeline([('Scaler',StandardScaler()),('AB',AdaBoostRegressor())])
# ensembles['ScaledAB-RFR'] = Pipeline([('Scaler',StandardScaler()),('AB-RFR',AdaBoostRegressor(base_estimator=RandomForestRegressor(n_estimators=50)))])
# ensembles['ScaledAB-CART'] = Pipeline([('Scaler',StandardScaler()),('AB-CART',AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=10)))])
# ensembles['ScaledCART'] = Pipeline([('Scale',StandardScaler()),('CART',DecisionTreeRegressor())])
# result =[]
# for key in ensembles:
#     kFold = KFold(n_splits=num_folds,random_state=seed)
#     cv_result = cross_val_score(ensembles[key],X_train,Y_train,cv=kFold,scoring=scoring)
#     result.append(cv_result)
#     print('%s: %f (%f)'%(key,cv_result.mean(),cv_result.std()))
    #结果：ScaledAB: -8946.815544 (1851.704966)
            # ScaledAB-RFR: -2399.576290 (105.623486)
            # ScaledAB-CART: -2700.239167 (139.193660)
            # ScaledCART: -5208.511684 (0.417854)
#集成算法调整参数
caler = StandardScaler().fit(X_train)
rescaledX = caler.transform(X_train)
param_grid = {'n_estimators':[60,70,80]}
model = AdaBoostRegressor(DecisionTreeRegressor())
kfold = KFold(n_splits=num_folds,random_state=seed)
grid = GridSearchCV(estimator=model,param_grid=param_grid,scoring=scoring,cv=kfold)
grid_result = grid.fit(X=rescaledX,y = Y_train)
print('最优：%s 使用%s'% (grid_result.best_score_,grid_result.best_params_))
