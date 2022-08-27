import cv2
import numpy as np
import os

directory = 'E:\BigBigData'

for filename in os.listdir(directory):
    for i in range(0, 3):
        f = os.path.join(directory, filename)

        b = np.load(f)
        g = b['rgb.npy']

        size = 450, 850
        duration = 40
        fps = 25
        out = cv2.VideoWriter('E:/SmallSmallData/'+str(filename[0:len(filename)-4])+'_'+str(i + 1)+'.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
        for _ in range(int(fps * duration * i/3), int(fps * duration * (i+1)/3)):
            data = g[_, :, :, 0:3].reshape((size[0], size[1], 3))
            out.write(data)
        out.release()
