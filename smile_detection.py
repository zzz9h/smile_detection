# use the saved model
from sklearn.externals import joblib

import ML_ways_sklearn

import dlib         # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 numpy
import cv2          # 图像处理的库 OpenCv


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib_model/shape_predictor_68_face_landmarks.dat')

# OpenCv 调用摄像头
cap = cv2.VideoCapture(0)

# 设置视频参数
cap.set(3, 480)
  # 待会要写的字体
font = cv2.FONT_HERSHEY_SIMPLEX

def get_features(img_rd):

    # 输入:  img_rd:      图像文件
    # 输出:  positions_lip_arr:  feature point 49 to feature point 68, 20 feature points / 40D in all

    # 取灰度
    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    # 计算68点坐标
    positions_68_arr = []
    faces = detector(img_gray, 0)
    landmarks = np.matrix([[p.x, p.y] for p in predictor(img_rd, faces[0]).parts()])

    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])
        positions_68_arr.append(pos)
        # 特征点画圈
        cv2.circle(img_rd, pos, 2, color=(255, 255, 255))

    positions_lip_arr = []
    # 将点 49-68 写入 CSV
    # 即 positions_68_arr[48]-positions_68_arr[67]
    for i in range(48, 68):
        positions_lip_arr.append(positions_68_arr[i][0])
        positions_lip_arr.append(positions_68_arr[i][1])

    return positions_lip_arr


while cap.isOpened():
    # 480 height * 640 width
    flag, img_rd = cap.read()
    key = cv2.waitKey(1)

    img_gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)

    # 人脸数 faces
    faces = detector(img_gray, 0)
    # 检测到人脸
    if len(faces) != 0:
        # 提取单张40维度特征
        positions_lip_test = get_features(img_rd)

        # path of models
        path_models = "data/data_models/"
        # #########  MLPC  ###########
        MLPC = joblib.load(path_models+"model_MLPC1.m")
        ss_MLPC = ML_ways_sklearn.model_MLPC()
        X_test_MLPC = ss_MLPC.transform([positions_lip_test])
        y_predict_MLPC = str(MLPC.predict(X_test_MLPC)[0]).replace('0', "no smile").replace('1', "with smile")
        if y_predict_MLPC=="with smile":
            im_rd=cv2.putText(img_rd, "smile", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            im_rd=cv2.putText(img_rd, "no smile", (20, 50), font, 1, (0, 0, 0), 1, cv2.LINE_AA)


        print('\n')
            # 按下 's' 键保存
        if key == ord('s'):
            ss_cnt += 1
            print(path_screenshots + "ss_" + str(ss_cnt) + "_" +
                time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg")
            cv2.imwrite(path_screenshots + "ss_" + str(ss_cnt) + "_" +
                        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".jpg", img_rd)

        # 按下 'q' 键退出
        if key == ord('q'):
            break
        
    im_rd = cv2.putText(img_rd, "'S': screen shot", (20, 400), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    im_rd = cv2.putText(img_rd, "'Q': quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
    # 窗口显示
    # cv2.namedWindow("camera", 0) # 如果需要摄像头窗口大小可调
    cv2.imshow("camera", img_rd)

# 释放摄像头
cap.release()

# 删除建立的窗口
cv2.destroyAllWindows()
