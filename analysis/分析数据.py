#-*- encoding: utf-8 -*-
import matplotlib.pyplot as plt
import pandas as pd
'''
本函数是对数据属性的分析
'''
# fig = plt.figure()
# fig.set(alpha=0.2) #设定图表的颜色
#
# data_train = pd.read_csv(r"C:\Users\wangyujie\Desktop\\test\dataset\dataset\\train.csv")

#plt.subplotgrid((2,3),(0,0))
#统计数据的季节数量，结果是基本一致
# data_train.season.value_counts().plot(kind='bar') #柱状图
# plt.title(u'pm大小')
# plt.ylabel(u"数量")

#按照年份查看记录的年份分布 发现2015年比较多  依次递减 相差不是很大
# data_train.year.value_counts().plot(kind='bar') #柱状图
# plt.ylabel(u"测试数据中每年的数据条数")  # 设定纵坐标名称
# plt.xlabel(u'年份')
# plt.grid(b=True, which='major', axis='y')
# plt.title(u"查看每年记录多少")

# plt.subplot2grid((2,3),(0,1))
#年份和pm大小的关系
# plt.scatter(data_train.year,data_train.PM_US_Post,)
# plt.ylabel(u"年份")
# plt.grid(b=True, which='major', axis='y')
# plt.title(u"pm的大小")

#月份和pm大小的关系（还是关系很大的）
# plt.scatter(data_train.month,data_train.PM_US_Post)
# plt.ylabel(u'月份')
# plt.grid(b=True,which='major' ,axis='y')
# plt.title(u'pm的大小和月份的关系')

#日期和pm的关系（还是有关系的）
# plt.scatter(data_train.day,data_train.PM_US_Post)
# plt.ylabel(u'日期')
# plt.grid(b=True,which='major' ,axis='y')
# plt.title(u'pm的大小和日期的关系')


#温度和pm大小的关系
# plt.scatter(data_train.TEMP,data_train.PM_US_Post)
# plt.ylabel(u'温度')
# plt.title(u'温度和pm的关系')

#小时和pm大小的关系
# data_train.hour.value_counts().plot(kind='bar')# 柱状图
# plt.title(u"按照小时查看pm") # 标题plt.scatter(data_train.hour,data_train.PM_US_Post)
# plt.xlabel(u'小时')
# plt.title(u'小时和pm的关系')
# plt.xlabel(u"小时")
# plt.ylabel(u'数量')


#季节和pm大小的关系（发现季节与数量没关系）
# data_train.season.value_counts().plot(kind='bar')# 柱状图
# plt.title(u"按照季节查看pm")  # 标题
# plt.scatter(data_train.season,data_train.PM_US_Post)
# plt.xlabel(u'季节')
# plt.title(u'季节和pm密度的关系')
# plt.xlabel(u"季节")
# plt.ylabel(u'密度大小')


#东四和pm大小的关系（）
# data_train.PM_Dongsi.value_counts().plot(kind='bar')# 柱状图
# plt.title(u"按照东四地区查看pm")  # 标题
# plt.scatter(data_train.season,data_train.PM_US_Post)
# plt.xlabel(u'季节')
# plt.title(u'季节和pm密度的关系')
# plt.xlabel(u"季节")
# plt.ylabel(u'密度大小')


#露点温度和pm大小的关系（）
# data_train.DEWP.value_counts().plot(kind='bar')# 柱状图
# plt.title(u"按照露点温度查看pm")  # 标题
# plt.ylabel(u'数量')
# plt.scatter(data_train.DEWP,data_train.PM_US_Post)
# plt.xlabel(u'露点温度')
# plt.title(u'露点温度和pm密度的关系')
# plt.xlabel(u"露点温度")
# plt.ylabel(u'密度大小')

#湿度（HUMI）和pm的密度大小关系
# plt.scatter(data_train.HUMI,data_train.PM_US_Post)
# plt.xlabel(u'湿度')
# plt.title(u'湿度和pm密度的关系')
# plt.ylabel(u'pm密度大小')

#气压（PRES）和pm的密度大小关系
# plt.scatter(data_train.PRES,data_train.PM_US_Post)
# plt.xlabel(u'气压')
# plt.title(u'气压和pm密度的关系')
# plt.ylabel(u'pm密度大小')

#温度（TEMP）和pm的密度大小关系
# plt.scatter(data_train.PRES,data_train.PM_US_Post)
# plt.xlabel(u'温度')
# plt.title(u'温度和pm密度的关系')
# plt.ylabel(u'pm密度大小')

# #风向（cbwd）和pm的密度大小关系
# plt.scatter(data_train.cbwd,data_train.PM_US_Post)
# plt.xlabel(u'风向')
# plt.title(u'风向和pm密度的关系')
# plt.ylabel(u'pm密度大小')

#累计风速（Iws）和pm的密度大小关系
# plt.scatter(data_train.Iws,data_train.PM_US_Post)
# plt.xlabel(u'累计风速')
# plt.title(u'累计风速和pm密度的关系')
# plt.ylabel(u'pm密度大小')

#累计风速（Iws）和pm的密度大小关系
# plt.scatter(data_train.Iws,data_train.PM_US_Post)
# plt.xlabel(u'累计风速')
# plt.title(u'累计风速和pm密度的关系')
# plt.ylabel(u'pm密度大小')

#累计风速和密度大小的关系
# plt.scatter(data_train.Iws,data_train.PM_US_Post)
# plt.title(u'累计风速和密度大小的关系')
# plt.xlabel(u'累计风速')
# plt.ylabel(u'pm密度大小')
#由于有空值，所以没法对应计算
# data_train.PM_US_Post[data_train.PM_Dongsi == 1].plot(kind='kde')
# data_train.PM_US_Post[data_train.PM_Dongsihuan == 2].plot(kind='kde')
# data_train.PM_US_Post[data_train.PM_Nongzhanguan == 3].plot(kind='kde')
# plt.xlabel(u"pm的重量")# plots an axis lable
# plt.ylabel(u"密度")
# plt.title(u"三个地区的pm")
# plt.legend((u'1东四', u'2东四环',u'3南东环'),loc='best') # sets our legend for our graph.



# plt.show()
