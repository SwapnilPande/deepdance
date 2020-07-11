import cv2
from tqdm import tqdm
import numpy as np


path1 = 'video/aligned-caroline-choreo.mp4'
path2 = 'video/aligned-caroline-choreo-2.mp4'

cap1 = cv2.VideoCapture(path1)
cap2 = cv2.VideoCapture(path2)
fps = cap1.get(cv2.CAP_PROP_FPS)
frame_width1 = cap1.get(3)
frame_height1 = cap1.get(4)
frame_width2 = cap2.get(3)
frame_height2 = cap2.get(4)


while 1:
    success1, frame1 = cap1.read()
    success2, frame2 = cap2.read()
    if success1 and success2:
        vis = np.concatenate((frame1, frame2), axis=1)
        new_vis = cv2.resize(vis, (int(frame_width1), int(frame_height1/2)))
        cv2.imshow("synced frames", new_vis)
        cv2.waitKey(0)
    else:
        break
