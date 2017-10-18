# -*-coding:utf8-*-

from __future__ import print_function
import sys
import codecs
import numpy as np

class Data_Loader(object):
    # 一些属性写在这里
    def __init__(self, train_path='../hxx/tr_all',
                validation_path='../hxx/tv_all',
                new_train_path='../new_data/tr.txt',
                new_validation1_path='../new_data/tv_1.txt',
                new_validation2_path='../new_data/tv_2.txt'):

        self.train_path = new_train_path
        self.validation_path = new_validation1_path
        self.test_path = new_validation2_path
        # TODO: 其他可能用到的属性，可以给默认的参数，也可以不给
        print("data loading...")
    
    # 数据加载 
    def load(self, file_path):
        # 输入：数据的地址
        # 输出：原组，格式如下：(特征的list，标签的list)
        # TODO：按照方法的功能和输入输出进行完善
	feature=[]
	label=[]

        with codecs.open(file_path,'r', encoding='utf8') as fp:
            for line in fp.readlines():
		l_feature=[]
		l_label=[]
		stringdata=line.strip().split()
		floatdata=[]
                floatdata = map(eval, stringdata)
		# for xString in stringdata:
		#     floatdata.append(float(xString))
		# l_feature=floatdata[7:25]
              	feature.append(floatdata[7:-1])
		# l_label=floatdata[-1]
		label.append(floatdata[-1])
        print("load", len(feature), "data")
	return np.array(feature), np.array(label)
    
    # 数据加载 
    def new_load(self, file_path):
        # 输入：数据的地址
        # 输出：原组，格式如下：(特征的list，标签的list)
        # TODO：按照方法的功能和输入输出进行完善
	feature=[]
	label=[]
	#r_std=np.load('../new_data/std.npz')

        with codecs.open(file_path,'r', encoding='utf8') as fp:
            for line in fp.readlines():
		# l_feature=[]
		# l_label=[]
		stringdata=line.strip().split()
                floatdata = map(eval, stringdata)
		# for xString in stringdata:
		#     floatdata.append(float(xString))
		# l_feature=floatdata[7:25]
              	feature.append(floatdata[5:-1])
		# l_label=floatdata[-1]
		label.append(floatdata[-1])
	f = np.array(feature)
	#m_q = np.mean(f,axis=0)
	#s_q = np.std(f,axis=0)
	#np.savez("../new_data/std.npz",mq=m_q,sq=s_q)
	#f = (f-r_std["mq"])/r_std["sq"]
        print("load", len(feature), "data")
	return f, (np.array(label)*249+1)/10.0

# 测试数据加载模块
def test_load():
    data_loader = Data_Loader()
    feature, label = data_loader.new_load(data_loader.test_path) 
    print(len(feature), len(label))
    print(feature[:2], label[:2])

if __name__ == '__main__':
    test_load()
