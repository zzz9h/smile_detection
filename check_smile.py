# use the saved model
from sklearn.externals import joblib

from get_features import get_features
import ML_ways_sklearn

import cv2

# path of test img
path_test_img = "data/data_imgs/test_imgs/i064qa-mn.jpg"

# 提取单张40维度特征
positions_lip_test = get_features(path_test_img)

# path of models
path_models = "data/data_models/"

print("The result of"+path_test_img+":")
print('\n')

# #########  MLPC  ###########
MLPC = joblib.load(path_models+"model_MLPC.m")
ss_MLPC = ML_ways_sklearn.model_MLPC()
X_test_MLPC = ss_MLPC.transform([positions_lip_test])
y_predict_MLPC = str(MLPC.predict(X_test_MLPC)[0]).replace('0', "no smile").replace('1', "with smile")
print("MLPC:", y_predict_MLPC)



img_test = cv2.imread(path_test_img)

img_height = int(img_test.shape[0])
img_width = int(img_test.shape[1])

# show the results on the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img_test, "MLPC:  "+y_predict_MLPC, (int(img_height/10), int(img_width/10)*3), font, 0.8, (84, 255, 159), 1, cv2.LINE_AA)
cv2.namedWindow("img", 2)
cv2.imshow("img", img_test)
cv2.waitKey(0)
