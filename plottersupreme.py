import math

import matplotlib.pyplot as plt
import numpy as np

filepath = r'C:\Users\Youssef Ziad\PycharmProjects\SwinBERT\models\table1\vatex-carla-exp2\log\log.txt'

train_loss_arr = []
val_loss_arr = []
bleu4_arr = []
meteor_arr = []
cider_arr = []

with open(filepath, 'r') as file:
    lines = file.readlines()
    for line in lines:
        if line[42:46] == 'eta:':
            losspos = line.find('loss: ')
            cutline = line[losspos:losspos+25]
            train_loss_arr.append(float(line[losspos+6:losspos+12]))
            vlosspos = cutline.find('(')
            val_loss_arr.append(float(cutline[vlosspos+1:vlosspos+7]))
        elif line[42:61] == 'evaluation result: ':
            bleu4_pos_start = line.find('\'Bleu_4\': ')
            bleu4_pos_end = bleu4_pos_start+line[bleu4_pos_start:].find(',')
            bleu4_arr.append(float(line[bleu4_pos_start+10:bleu4_pos_end-1]))
            meteor_pos_start = line.find('\'METEOR\': ')
            meteor_pos_end = meteor_pos_start + line[meteor_pos_start:].find(',')
            meteor_arr.append(float(line[meteor_pos_start + 10:meteor_pos_end - 1]))
            cider_pos_start = line.find('\'CIDEr\': ')
            cider_pos_end = cider_pos_start + line[cider_pos_start:].find(',')
            cider_arr.append(float(line[cider_pos_start + 9:cider_pos_end - 1]))

meteor_arr = meteor_arr[1:]
bleu4_arr = bleu4_arr[1:]
cider_arr = cider_arr[1:]

print(bleu4_arr)

epoch = np.arange(0,13,(1/45))
epoch = np.ndarray.tolist(epoch)
# epoch = list(range(1,16))

fig1, ax1 = plt.subplots()

ax1.plot(epoch, train_loss_arr, marker='.')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.plot(epoch, val_loss_arr, marker='.')
plt.show()

