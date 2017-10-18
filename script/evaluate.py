# -*- coding:utf8 -*-

import numpy as np
from math import sqrt

f_target = open('./result/target.label', 'r')
f_dense = open('./result/result1.dense', 'r')
f_conv = open('./result/result1.conv', 'r')


target = f_target.readlines()
target = map(float, target)
target = np.array(target)

dense = f_dense.readlines()
dense = map(float, dense)
dense = np.array(dense)

conv = f_conv.readlines()
conv = map(float, conv)
conv = np.array(conv)

print sqrt(np.sum((target-dense)**2)/len(target))
print sqrt(np.sum((target-conv)**2)/len(target))
