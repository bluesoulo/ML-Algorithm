import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_squared_error
#ignore warnings
import warnings
warnings.filterwarnings('ignore')

map = [{'spring':0,'summer':1,'autumn':2,'winter':3},
       {'small':0,'medium':1,'large':2},
       {'low':0,'medium':1,'high':2}]
def read_y(filename):
    with  open(filename,'r',encoding='utf-8') as f:
        data = []
        for line in f:
            tem = line.strip().split()
            value = [eval(i) for i in tem]
            data.append(value)
    return data

def read_data(filename):
    with open(filename,'r',encoding='utf-8') as f:
        values = []
        for line in f:
            words = line.strip().split()
            value = []
            for i in range(len(words)):
                if i<3:
                    value.append(map[i][words[i]])
                elif words[i] != 'XXXXXXX' :
                        value.append(eval(words[i]))
                else:
                    value.append(np.nan)
            values.append(value)
    return values

def miss_values(X):
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(X)
    return imp

def split_sets(data):
    data = np.array(data)
    return data[:,0:11], data[:,11:]

def standard(X):
    std = StandardScaler()
    std.fit(X)
    return std
def processing():
    train_data = read_data('./data/train.txt')
    train_X, train_Y = split_sets(train_data)
    test_X = read_data('./data/test.txt')
    test_Y = read_y('./data/test_labels.txt')
    imp = miss_values(train_X)
    train_X = imp.transform(train_X)
    test_X = imp.transform(test_X)
    std = standard(train_X)
    std_train_X = std.transform(train_X)
    std_test_X = std.transform(test_X)
    std_train_X, std_test_X, = np.delete(std_train_X, 9, axis=1), np.delete(std_test_X, 9, axis=1)
    return std_train_X, train_Y, std_test_X, test_Y
    #return train_X ,train_Y, test_X, test_Y

def train():
    train_X, train_Y, test_X, test_Y = processing()
    #model = MultiOutputRegressor(linear_model.LinearRegression()).fit(train_X, train_Y)
    #model = MultiOutputRegressor(linear_model.Ridge()).fit(train_X, train_Y)
    #model = MultiOutputRegressor(AdaBoostRegressor(n_estimators=100,learning_rate=0.01,loss='square')).fit(train_X, train_Y)
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,max_depth=6,random_state=1,oob_score=True,max_features=5)).fit(train_X, train_Y)
    #model = MultiOutputRegressor(xgb.XGBRegressor()).fit(train_X, train_Y)
    #print_score(model,train_X,train_Y)
    #print_score(model,test_X,test_Y)
    return model

def train2():
    train_X, train_Y, test_X, test_Y = processing()
    train_X = np.concatenate((train_X,train_Y),axis=1)
    #model2 = MultiOutputRegressor(RandomForestRegressor(n_estimators=100,max_depth=6,random_state=1,oob_score=True,max_features=5)).fit(train_X, train_Y)
    model2 = MultiOutputRegressor(linear_model.LinearRegression()).fit(train_X, train_Y)
    model = train()
    print_score(model, test_X, test_Y)
    result = model.predict(test_X)
    test_X = np.concatenate((test_X,result),axis=1)
    print_score(model2, test_X, test_Y)


def print_score(model,X,Y):
    result = model.predict(X)
    result = np.array(result)
    result =np.maximum(result,0)
    Y = np.array(Y)
    print(mean_squared_error(Y, result))
    for i in range(0, 7):
        print(mean_squared_error(Y[:, i], result[:, i]))
    print('------------')

def change_parameter():
    train_X, train_Y, _, _ = processing()
    Y = np.array(train_Y)[:,0]
    param_test = {'max_depth':range(1,14,2)}
    gsearch1 = GridSearchCV(estimator = RandomForestRegressor(random_state=1),
                       param_grid = param_test, scoring='r2',cv=3)
    #gsearch1 = GridSearchCV(estimator=xgb.XGBRegressor(max_depth=1,random_state=0,learning_rate=0.05),param_grid=param_test, scoring='r2', cv=5)
    gsearch1.fit(train_X, Y)
    print( gsearch1.best_params_, gsearch1.best_score_)

def heat():#
    train_data = read_data('./data/train.txt')
    train_X, train_Y = split_sets(train_data)
    imp = miss_values(train_X)
    ticklabels = ['season', 'size', 'speed', 'max_PH', 'min_O2', 'avg_cl', 'avg_NO3', 'avg_NH4', 'avg1_PO4', 'avg2_PO4',
                  'avg_Chlorophyll']
    '''
    corr_X = np.corrcoef(np.array(imp.transform(train_X)).transpose())
    sns.heatmap(corr_X,vmax = 0.9, cmap = 'Reds', xticklabels=ticklabels,yticklabels=ticklabels,square=True)
    plt.savefig('./heatMap.png')'''
    corr_Y = np.corrcoef(np.array(train_Y).transpose())
    print(np.shape(corr_Y))
    sns.heatmap(corr_Y, vmax = 0.9, cmap='Reds',square=True)
    plt.show()

def handle_outliers():#画箱线图
    train_data = read_data('./data/train.txt')
    train_X, train_Y = split_sets(train_data)
    imp = miss_values(train_X)
    train_X = imp.transform(train_X)
    train_X = np.array(train_X)
    data = {}
    data['season'] = train_X[:,0]
    data['size'] = train_X[:,1]
    data['speed'] = train_X[:,2]
    data['max_PH'] = train_X[:,3]
    data['min_O2'] = train_X[:,4]
    data['avg_cl'] = train_X[:,5]
    data['avg_NO3'] = train_X[:,6]
    data['avg_NH4'] = train_X[:,7]
    data['avg1_PO4'] = train_X[:,8]
    data['avg2_PO4'] = train_X[:,9]
    data['avg_Chlorophyll'] = train_X[:,10]
    df = pd.DataFrame(data)
    print(df.describe())
    df.plot.box(title="Box-plot")
    plt.grid(linestyle="--", alpha=0.3)
    plt.show()

def hist():
    train_data = read_data('./data/train.txt')
    train_X, train_Y = split_sets(train_data)
    imp = miss_values(train_X)
    train_X = imp.transform(train_X)
    train_X = np.array(train_X).transpose()
    plt.figure(figsize=(30, 30), dpi=80)
    ticklabels = ['season', 'size', 'speed', 'max_PH', 'min_O2', 'avg_cl', 'avg_NO3', 'avg_NH4', 'avg1_PO4',
                  'avg2_PO4','avg_Chlorophyll']
    for i in range(11):
        ax1 = plt.subplot(4,3,i+1)
        plt.hist(train_X[i])
        ax1.set_title(ticklabels[i])
    plt.savefig('hist.png')
    plt.show()
if __name__ == '__main__':
   #change_parameter()#调参
   #train2()  #训练
   #hist()     #画直方图
   #heat()     #画相关系数热图
   #processing() #处理数据，得掉测试集和训练集
    handle_outliers()

