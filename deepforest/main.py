from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier
from sklearn import svm
from skimage import io,segmentation,measure
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

import func
original_image = io.imread('28689_sat.jpg')
label_image = io.imread('28689_mask.png')

# 进行超像素分割
segments = segmentation.slic(original_image, n_segments=1000, compactness=10)
dict_temp = func.superpixels_to_label(segments,label_image)
y = np.array(dict_temp)
X = func.get_superpixels_feature(image=original_image,segments=segments)



X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = CascadeForestClassifier(random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("\nTesting Accuracy: {:.3f} %".format(acc))


# 创建随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
nb_classifier = GaussianNB()
svm_classifier = svm.SVC()
# 训练分类器
rf_classifier.fit(X_train, y_train)
# 训练分类器
nb_classifier.fit(X_train, y_train)

svm_classifier.fit(X_train,y_train)
# 在测试集上进行预测
y_pred_rf = rf_classifier.predict(X_test)
y_pred_nb = nb_classifier.predict(X_test)
y_pred_svm = svm_classifier.predict(X_test)
# 计算准确率
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_nb = accuracy_score(y_test,y_pred_nb)
accuracy_svm = accuracy_score(y_test,y_pred_svm)

print("rf准确率:", accuracy_rf)
print("nb准确率", accuracy_nb)
print("svm准确率", accuracy_svm)
