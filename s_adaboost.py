# Adaboost算法思路：
# 1.基于样本权重相同的数据集建立一个弱分类器，计算弱分类器的误差，分类器权重和新的样本权重
# 2.基于新的样本权重的数据器建立一个新的弱分类器
# 3.循环1.2直到到达指定的M个分类器或者分类误差小于指定的阈值，线性组合弱分类器为一个强分类器。
