from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from deepforest import CascadeForestClassifier
from skimage import io,segmentation,measure
import numpy as np
# X, y = load_digits(return_X_y=True)
# print(X[0])
# 读取原始图像和标签图像
import func
original_image = io.imread('28689_sat.jpg')
label_image = io.imread('28689_mask.png')

# 进行超像素分割
segments = segmentation.slic(original_image, n_segments=100, compactness=10)
dict_temp = func.superpixels_to_label(segments,label_image)
y = np.array(dict_temp)
X = func.get_superpixels_feature(image=original_image,segments=segments)
# print(X[0])
# print(list_array.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
model = CascadeForestClassifier(random_state=1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100
print("\nTesting Accuracy: {:.3f} %".format(acc))
