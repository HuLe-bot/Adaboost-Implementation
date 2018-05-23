# Adaboost算法思路：
# 1.基于样本权重相同的数据集建立一个弱分类器，计算弱分类器的误差，分类器权重和新的样本权重
# 2.基于新的样本权重的数据器建立一个新的弱分类器
# 3.循环1.2直到到达指定的M个分类器或者分类误差小于指定的阈值，线性组合弱分类器为一个强分类器。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_hastie_10_2

# Helper function: get error rate
def get_error_rate(pred,Y):
    total = 0
    for (x, y) in zip(pred, Y):
        if x != y:
            total += 1
    return total/float(len(Y))

# Helper function :print error rate
def print_error_rate(err):
    print('Error rate: Training: %.4f - Test: %.4f',err)

# Helper function: generic classifier
def generic_clf(Y_train,X_train,Y_test,X_test,clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train,Y_train),get_error_rate(pred_test,Y_test)

# Adaboost Implementation
def adaboost_clf(Y_train,X_train,Y_test,X_test,M,clf):
    n_train,n_test = len(X_train) ,len(X_test)
    #Initialize weights
    w = np.ones(n_train) / n_train
    pred_train = [np.zeros(n_train)]
    pred_test = [np.zeros(n_test)]

    for i in range(M):
        # Fit a classifier with the specific weights
        clf.fit(X_train,Y_train,sample_weight = w)
        pred_train_i = clf.predict(X_train)   # Gm(x)
        pred_test_i = clf.predict(X_test)
        print('pred_train_i :',pred_train_i)
        print('pred_test_i :',pred_test_i)

        # Indicator function
        miss = [int(x) for x in (pred_train_i !=Y_train)]
        print('miss: ',miss)
        # Equivalent with 1/-1 to update weights

        miss2 = [x if x==1 else -1  for x in miss]  # correct:-1  mistake:1
        print('miss2:',miss2)
        # Error
        err_m = np.dot(w,miss)/sum(w) # sum(w)=1  作为Zm做归一化因子
        # Alpha
        alpha_m = 0.5 *np.log((1-err_m)/float(err_m))
        # New weights
        w = np.multiply(w,np.exp([float(x) * alpha_m for x in miss2]))
        print('w new:',w)
        # Add to prediction G(x)
        pred_train = [sum(x) for x in zip(pred_train,[x * alpha_m for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test,[x *alpha_m for x in pred_test_i])]
        pred_train, pred_test = np.sign(pred_train), np.sign(pred_test)

        return get_error_rate(pred_train,Y_train),get_error_rate(pred_test,Y_test)
# Plot function
def plot_error_rate(er_train,er_test):
    df_error = pd.DataFrame([er_train,er_test]).T
    df_error.columns = ['Training','Test']
    plot1 = df_error.plot(linewidth = 3,figsize = (8,6),color = ['ligthblue','darkblue'],grid = True)
    plot1.set_xlabel('Number of iterations',fontsize=12)
    plot1.set_xticklabels(range(0,450,50))
    plot1.set_ylabel('Error rate',fontsize=12)
    plot1.set_title('Error rate vs number of iterations',fontsize = 16)
    plt.axhline(y = er_test[0],linewidth = 1, color = 'red',ls='dashed')

# main script
if __name__ =='__main__':
    # Read data
    x,y = make_hastie_10_2()
    df = pd.DataFrame(x)
    df['Y'] = y

    # Split into training and test set
    train ,test = train_test_split(df,test_size=0.2)
    X_train, Y_train = train.ix[:,:-1],train.ix[:,-1]
    X_test , Y_test = test.ix[:,:-1],test.ix[:,-1]

    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier(max_depth=1,random_state=1)
    er_tree = generic_clf(Y_train,X_train,Y_test,X_test,clf_tree)

    # Fit Adaboost classifier using a decision tree as base estimator
    # Tree with different number of iterations
    er_train,er_test = [er_tree[0]],[er_tree[1]]
    x_range = range(10,410,10)
    for i in x_range:
        er_i = adaboost_clf(Y_train,X_train,Y_test,X_test,i,clf_tree)
        er_train.append(er_i[0])
        er_test.append(er_i[1])
    # Compare error rate vs number of iteration
    plot_error_rate(er_train,er_test)

