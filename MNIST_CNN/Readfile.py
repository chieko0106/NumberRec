# -*- coding: UTF-8 -*-
import os
import csv

def get_data(name,ifsig):
    father_path=os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")
    test_data_path = father_path+'/data/'+name
    test_data = {}
    with open(test_data_path,'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for line in spamreader:
            line = line[0]
            line = line.split(',')
            if ifsig == True:
                sig = line[0]
                line.pop(0)
            else:
                sig = ifsig
            if sig not in test_data.keys():
                test_data[sig] = []
            test_data[sig].append(line)
    return(test_data)


test = get_data('test.csv','test')
test = test['test']
test.pop(0)
i = 0
current = test[109]
for ii in range(len(current)):
    current[ii] = int(current[ii])

import numpy as np
data = np.matrix(current,dtype='float')
new_data = np.reshape(data,(28,28))

from PIL import Image
new_im = Image.fromarray(new_data.astype(np.uint8))

import matplotlib.pyplot as plt
plt.imshow(new_data, cmap=plt.cm.gray, interpolation='nearest')
new_im.show()


