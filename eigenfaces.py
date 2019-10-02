#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 10:57:50 2019

@author: avnish
"""
import pandas as pd
import numpy as np
from scipy import misc
from matplotlib import pylab as plt
import matplotlib.cm as cm
#matplotlib inline

train_labels, train_data = [], []
for line in open('./faces/train.txt'):
    im = misc.imread(line.strip().split()[0])
    train_data.append(im.reshape(2500,))
    train_labels.append(line.strip().split()[1])

train_data, train_labels = np.array(train_data, dtype=float), np.array(train_labels, dtype=int)

print train_data.shape, train_labels.shape

plt.imshow(train_data[10, :].reshape(50,50), cmap = cm.Greys_r)
plt.show()

ones_array = np.ones(540)

added_columns = np.matmul(ones_array,train_data)

average_face = np.true_divide(added_columns,540)


plt.imshow(average_face.reshape(50,50), cmap = cm.Greys_r)
plt.show()

