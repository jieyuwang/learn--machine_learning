#-*- encoding: utf-8 -*-
from sklearn import linear_model
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge#岭回归
from sklearn.linear_model import LinearRegression# 线性回归
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier#随机森林 比线性回归好点
from  sklearn.linear_model import Lasso
from  sklearn.linear_model import ElasticNet
from  sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor #GDBT集成模型
from sklearn.ensemble import AdaBoostRegressor #提升模型
from sklearn.ensemble import AdaBoostClassifier #提升模型
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor,MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

from sklearn.ensemble import GradientBoostingRegressor #随机梯度提升
from sklearn.tree import DecisionTreeRegressor#决策树
from sklearn.tree import ExtraTreeRegressor#极端随机树
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier#knn近邻模型
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
def deal_data_test(filename):
    '''对数据的预处理函数:
        1.把后面几个缺值属性用中值进行补全，删除no属性
        2.把风向映射为int类型 {'NW': 1, 'SE': 2, 'NE': 3, 'cv': 4}
        3.有三个地区属性缺失值很多，所以进行了补全和预测，预测函数在在另外的函数
    '''
    dataTrain = pd.read_csv(filename)
    dataTrain = dataTrain.sort_values(by=['year', 'month', 'day', 'hour'])
    # 对训练集进行简单处理,对于少数缺值的用均值补全
    dataTrain.DEWP.fillna(method='ffill', limit=1, inplace=True)
    dataTrain.HUMI.fillna(method='ffill', limit=240, inplace=True)
    dataTrain.PRES.fillna(method='ffill', limit=240, inplace=True)
    dataTrain.TEMP.fillna(method='ffill', limit=1, inplace=True)

    # 采用出现最频繁的值填充,并把str转为int
    freq_port = dataTrain.cbwd.dropna().mode()[0]
    dataTrain['cbwd'].fillna(freq_port, inplace=True)
    # 把风向转化为1,2,3,4 f方便放入模型
    cbwd_mapping = {'NW': 1, 'SE': 2, 'NE': 3, 'cv': 4}
    dataTrain['cbwd'] = dataTrain['cbwd'].map(cbwd_mapping)
    dataTrain.Iws.fillna(dataTrain.Iws.mean(), inplace=True)
    #dataTrain['Iws'] = dataTrain['Iws'] / max(dataTrain.Iws)
    dataTrain.precipitation.fillna(dataTrain.precipitation.mean(), inplace=True)
    dataTrain.Iprec.fillna(dataTrain.Iprec.mean(), inplace=True)
    # 三个地区互相赋值，最后形成训练集，一共有18958条记录
    dataTrain['PM_Dongsi'] = dataTrain['PM_Dongsi'].fillna((dataTrain['PM_Dongsihuan']))
    dataTrain['PM_Dongsi'] = dataTrain['PM_Dongsi'].fillna(dataTrain['PM_Nongzhanguan'])
    dataTrain['PM_Dongsihuan'] = dataTrain['PM_Dongsihuan'].fillna((dataTrain['PM_Dongsi']))
    dataTrain['PM_Dongsihuan'] = dataTrain['PM_Dongsihuan'].fillna(dataTrain['PM_Nongzhanguan'])
    dataTrain['PM_Nongzhanguan'] = dataTrain['PM_Nongzhanguan'].fillna((dataTrain['PM_Dongsi']))
    dataTrain['PM_Nongzhanguan'] = dataTrain['PM_Nongzhanguan'].fillna(dataTrain['PM_Dongsihuan'])
    # 排序,将三个地区有值的放在前面，方便切出来当成训练集，剩下的为测试集。
    dataTrain = dataTrain.sort_values(by=['PM_Dongsi'])
    # 对测试集划分训练集和测试集
    print(dataTrain.info())
    data3_train = dataTrain[0:6308]
    data3_test = dataTrain[6308:12597]
    return data3_train,data3_test
def deal_data_train(filename):
    '''对数据的预处理函数:
        1.把后面几个缺值属性用中值进行补全，删除no属性
        2.把风向映射为int类型 {'NW': 1, 'SE': 2, 'NE': 3, 'cv': 4}
        3.有三个地区属性缺失值很多，所以进行了补全和预测，预测函数在在另外的函数
    '''
    dataTrain = pd.read_csv(filename)
    dataTrain = dataTrain.sort_values(by=['year', 'month', 'day', 'hour'])
    # 对训练集进行简单处理,对于少数缺值的用均值补全
    dataTrain.DEWP.fillna(method='ffill', limit=10, inplace=True)
    dataTrain.HUMI.fillna(method='ffill', limit=240, inplace=True)
    dataTrain.PRES.fillna(method='ffill', limit=240, inplace=True)
    dataTrain.TEMP.fillna(method='ffill', limit=1, inplace=True)
    # 采用出现最频繁的值填充,并把str转为int
    freq_port = dataTrain.cbwd.dropna().mode()[0]
    dataTrain['cbwd'].fillna(freq_port, inplace=True)
    # 把风向转化为1,2,3,4 f方便放入模型
    cbwd_mapping = {'NW': 1, 'SE': 2, 'NE': 3, 'cv': 4}
    dataTrain['cbwd'] = dataTrain['cbwd'].map(cbwd_mapping)
    dataTrain.Iws.fillna(dataTrain.Iws.dropna().mode()[0], inplace=True)
    #dataTrain['Iws'] = dataTrain['Iws'] / max(dataTrain.Iws)
    dataTrain.precipitation.fillna(dataTrain.precipitation.dropna().mode()[0], inplace=True)
    dataTrain.Iprec.fillna(dataTrain.Iprec.dropna().mode()[0], inplace=True)
    # 三个地区互相赋值，最后形成训练集，一共有18958条记录
    dataTrain['PM_Dongsi'] = dataTrain['PM_Dongsi'].fillna((dataTrain['PM_Dongsihuan']))
    dataTrain['PM_Dongsi'] = dataTrain['PM_Dongsi'].fillna(dataTrain['PM_Nongzhanguan'])
    dataTrain['PM_Dongsihuan'] = dataTrain['PM_Dongsihuan'].fillna((dataTrain['PM_Dongsi']))
    dataTrain['PM_Dongsihuan'] = dataTrain['PM_Dongsihuan'].fillna(dataTrain['PM_Nongzhanguan'])
    dataTrain['PM_Nongzhanguan'] = dataTrain['PM_Nongzhanguan'].fillna((dataTrain['PM_Dongsi']))
    dataTrain['PM_Nongzhanguan'] = dataTrain['PM_Nongzhanguan'].fillna(dataTrain['PM_Dongsihuan'])
    # 排序,将三个地区有值的放在前面，方便切出来当成训练集，剩下的为测试集。
    dataTrain = dataTrain.sort_values(by=['PM_Dongsi'])
    # #对训练集划分训练集和测试集
    print(dataTrain.info())
    data3_train = dataTrain[0:18958]
    data3_test = dataTrain[18958:37791]
    return data3_train, data3_test
def predata(train_1,test_1,filename):
    '''
       对雾霾三种地方（由于数据不全）进行预测，补全后的训练集有18958条记录，测试集为37790-18958。
       对集合进行预测，并最终生成文件train_deal.csv文件
       :param train_1: 训练集
       :param test_1:测试集
       :return: 返回最终处理过的没有任何缺失的数据
       '''
    # train_1, test_1 = train_test_split(train, train_size=0.7)
    #把目标集和需要预测的集合拿出来
    train_PM_Dongsi = train_1.pop('PM_Dongsi')
    train_PM_Dongsihuan = train_1.pop('PM_Dongsihuan')
    train_PM_Nongzhanguan = train_1.pop('PM_Nongzhanguan')
    # train_pre = train_1.pop('PM_US_Post')

    test_PM_Dongsi = test_1.pop('PM_Dongsi')
    test_PM_Dongsihuan = test_1.pop('PM_Dongsihuan')
    test_PM_Nongzhanguan = test_1.pop('PM_Nongzhanguan')
    # test_pre = test_1.pop('PM_US_Post')

    cols = ['year', 'month', 'day', 'hour', 'season', 'DEWP',
            'HUMI', 'PRES', 'TEMP', 'cbwd', 'Iws', 'precipitation', 'Iprec','PM_US_Post']
    reg11 = DecisionTreeRegressor()
    reg = AdaBoostRegressor(reg11)
    reg.fit(train_1[cols],train_PM_Dongsihuan)
    pre_PM_Dongsihuan = reg.predict(test_1[cols])

    reg1 = DecisionTreeRegressor()
    reg = AdaBoostRegressor(reg1,n_estimators=60)
    reg.fit(train_1[cols], train_PM_Dongsi)
    pre_PM_Dongsi = reg.predict(test_1[cols])

    reg1= DecisionTreeRegressor()
    reg = AdaBoostRegressor(reg1,n_estimators=60)
    reg.fit(train_1[cols], train_PM_Nongzhanguan)
    pre_PM_Nongzhanguan = reg.predict(test_1[cols])


    test_1['PM_Dongsi'] = pre_PM_Dongsi
    train_1['PM_Dongsi'] = train_PM_Dongsi
    test_1['PM_Dongsihuan'] = pre_PM_Dongsihuan
    train_1['PM_Dongsihuan'] = train_PM_Dongsihuan
    test_1['PM_Nongzhanguan'] = pre_PM_Nongzhanguan
    train_1['PM_Nongzhanguan'] = train_PM_Nongzhanguan
    # #合并训练接和测试集
    train = pd.concat([test_1,train_1])
    # #生成最终处理后的文件tran_deal.csv文件
    train = train.sort_values(by=['No'])
    train.to_csv(filename,index=False)
def preFinall(train_filename,test_filename):
    train_ = pd.read_csv(train_filename)
    test_ = pd.read_csv(test_filename)

    train_PM_US_Post = train_.pop('PM_US_Post')
    col = ['year', 'month', 'day', 'hour']
    #开始训练
    reg1  = DecisionTreeRegressor()
    reg = AdaBoostRegressor(reg1)
    reg.fit(train_[col],train_PM_US_Post)
    pred_PM_US_Post = reg.predict(test_[col])
    test_['PM_US_Post'] =pred_PM_US_Post
    test_ =test_.sort_values(by='No')
    return test_
def trainNoNull(train,test):
    '''用于在本地对train的拆分训练，寻找最优的算法'''
    cols = ['PM_Dongsi','PM_Dongsihuan','PM_Nongzhanguan','year', 'month', 'day', 'hour', 'season',
            'DEWP','HUMI', 'PRES', 'TEMP', 'cbwd', 'Iws', 'precipitation', 'Iprec']
    reg = RandomForestRegressor()
    reg = AdaBoostRegressor(reg)#加上决策树是0.8332 加上随即森林是0.8148
    train_pre = train['PM_US_Post']
    reg.fit((train[cols]),train_pre)
    test__pred_PM_US_Post = reg.predict((test[cols]))
    return test__pred_PM_US_Post
if __name__ == '__main__':
    #由于属性中三个地区缺失的很多，所以我用其他属性进行了预测
    # filename_1 = 'train_deal.csv'#处理训练数据后的保存路径
    # filename_2 = 'test_deal.csv'#处理测试数据后的保存路径
    # #以下三行是处理雾霾三个地区缺失值和补全部分属性，最后生成train_deal文件用作目标文件.因为三个地区预测用到了算法，所以也需要划分训练测试集
    # #注意 因为训练和测试数据集长度不一致，所以需要到deal_data函数中注释掉一部分内容
    # data3_train, data3_test = deal_data_train(r'train.csv')
    # predata(data3_train, data3_test, filename_1)

    #通过把有三个地区值进行平均赋值给目标值，其他的进行预测，分数最高
    test_train, test_test = deal_data_test(r'test.csv')
    train_train, train_test = deal_data_train(r'train.csv')
    test_train['PM_US_Post'] = trainNoNull(train_train, test_train)
    test_train =test_train.sort_values(by = 'No')
    test_train[['No','PM_US_Post']].to_csv('new_train.csv',index=False)
    # # 根据测试好的算法进行预测
    test_ = preFinall('train.csv', 'split_test2.csv')
    test_ = test_.sort_values(by='No')
    train11 = pd.concat([test_, test_train])
    # # # # 生成最终处理后的文件tran_deal.csv文件
    sample = pd.read_csv('example.csv')
    train11 = train11.sort_values(by=['No'])
    sample['PM_US Post'] = np.array(train11.PM_US_Post)
    sample.to_csv('my_zuixin.csv', index=False)


