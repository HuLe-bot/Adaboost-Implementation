# use sklearn.ensemble.GradientBoostingClassifier

from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier

iris = load_iris()
clf = AdaBoostClassifier(n_estimators=100,learning_rate=1.0,random_state=0)  # 100个弱分类器,默认情况下弱学习器是树桩（根据单个特征生成的决策树）,可通过base_setimator进行调整
scores = cross_val_score(clf,iris.data,iris.target)
print(scores.mean())
print(iris)




