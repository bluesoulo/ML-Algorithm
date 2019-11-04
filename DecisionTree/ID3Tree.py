import numpy as np
import pandas as pd
import math
from sklearn import tree
class Node:
    def __init__(self,is_leaf,attribute,label,ls,values):
        self.is_leaf = is_leaf
        self.attribute = attribute
        self.label = label
        self.ls = ls
        self.values = values
        self.son = []
    def add_node(self,node):
        self.son.append(node)
def read_data():
    f = open('car.data','r',encoding='utf-8')
    data = []
    for line in f:
        tem = line.strip().split(',')
        if len(tem) >2:
            data.append(tem)
    data = pd.DataFrame(data)
    data.columns = ['buying','maint','doors','persons','lug_boot','safety','label']
    return data

def data_map(data):
    buying_map = {'low':3,'med':2,'high':1,'vhigh':0}
    doors_map = {'2':0,'3':1,'4':2,'5more':3}
    persons_map = {'2':0,'4':1,'more':2}
    lug_boot_map = {'small':0,'med':1,'big':2}
    safety_map ={'low':0,'med':1,'high':2}
    ls = [buying_map,buying_map,doors_map,persons_map,lug_boot_map,safety_map,safety_map]
    clo = data.columns
    for i in range(len(ls)):
        data[clo[i]] =data[clo[i]].map(ls[i])
    return data
def get_best_attribute(data):
    columns = data.columns
    labels = list(set(data[columns[-1]]))
    labels_map ={}
    for i in range(len(labels)):  #类别映射到one-hot变量,帮助计算个数。
        one_hot = [0]*len(labels)
        one_hot[i] = 1
        labels_map[labels[i]] = one_hot
    infor_entropys =[]
    for i in range(len(columns)-1):
        nums = {}
        for x,y in zip(data[columns[i]],data[columns[-1]]):
            if x not in nums.keys():
                nums[x] = labels_map[y]
            else:
                nums[x] = np.array(nums[x])+np.array(labels_map[y])
        informataion_entropy = 0
        for key in nums.keys():
            p = np.sum(nums[key]) / len(data[columns[i]])
            informataion_entropy = informataion_entropy  - math.log(p,2) * p
        infor_entropys.append(informataion_entropy)
    index = np.array(infor_entropys).argmin()
    return columns[index]

def is_one_class(data):
    if len(set(data['label']))>1:
        return False
    else:
        return True
def split_table(data,attribute):
    nums = data[attribute].unique()
    table_ls =[data[data[attribute].isin([i])] for i in nums]
    for table in table_ls:
        del table[attribute]
    return table_ls

def train(data, fnode,values=None):
    if is_one_class(data):
        label = list(set(data['label']))[0]
        #print(label,222)
        fnode.add_node(Node(True,None,label,None,values))
        return fnode
    else:
        attribute = get_best_attribute(data)
        table_ls = split_table(data,attribute)
        nums_ls = data[attribute].unique()
        node = Node(False,attribute,None,nums_ls,values)
        #print(node.attribute,node.values)
        fnode.add_node(node)
        for i in range(len(node.ls)):
            node = train(table_ls[i],node,nums_ls[i])
        return fnode
def print_tree(fnode):
    for node in fnode.son:
        if not node.is_leaf:
            print(node.is_leaf,node.attribute,node.values,node.ls,node.label)
        print_tree(node)

def for_predict(fnode,test_data):
    while(not fnode.is_leaf):
        attribute = fnode.attribute
        num = test_data[attribute][0]
        index = 0
        for j in range(len(fnode.ls)):
            if num == fnode.ls[j]:
                index = j
        fnode = fnode.son[index]
    return fnode.label
def test_to_table(test_data):
    test_data = pd.DataFrame([test_data])
    test_data.columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
    return test_data
if __name__ == '__main__':
    data = read_data()
    m =get_best_attribute(data)
    fnode = Node(False,None,None,None,None)
    fnode = train(data,fnode)
    test_data = ['vhigh','med','3','4','big','high']
    test_table = test_to_table(test_data)
    print(for_predict(fnode.son[0],test_table))
