import cv2
import numpy as np
import os

movement = ['waits at traffic light', 'crosses the street', 'walks on the sidewalk']
pedestrians = ['other pedestrians at close proximity', 'no other pedestrians at close proximity']
vehicles = ['vehicle on the left', 'vehicle on the right', 'no other vehicles approaching']

directory = 'E:\BigBigData'

json_string = "["

for filename in os.listdir(directory):

    b = np.load(directory+'/'+filename)

    rgb = b['rgb.npy']
    obs = b['obs.npy']
    act = b['action.npy']

    passed = False

    for q in range(0, 3):

        currrgb = rgb[int((q / 3) * rgb.shape[0]):int(((q + 1) / 3) * rgb.shape[0])]
        currobs = obs[int((q / 3) * obs.shape[0]):int(((q + 1) / 3) * obs.shape[0])]
        curract = act[int((q / 3) * act.shape[0]):int(((q + 1) / 3) * act.shape[0])]

        json_string += '{"videoID": "' + filename[0:len(filename)-4] + '_' + str(q + 1) + '", "enCap": ["'

        print(currobs.shape)

        happening_info = np.zeros((3, currobs.shape[0]), dtype='uint8')

        for i in range(currobs.shape[0]):
            if (curract[i, 0] == 1) & (curract[i, 1] == 0) & (passed is False):
                happening_info[0, i] = 1
            elif curract[i, 0] == 1:
                happening_info[0, i] = 2
            else:
                happening_info[0, i] = 0

            if i != 0:
                if (passed is False) & (happening_info[0, i] != 1) & (happening_info[0, i-1] == 1):
                    passed = True

            if np.count_nonzero(currobs[i] == 4) > 2000:
                happening_info[1, i] = 0
            else:
                happening_info[1, i] = 1

            carthresh = 4000

            right = np.count_nonzero(currobs[i, :, 400:] == 10) > carthresh
            left = np.count_nonzero(currobs[i, :, :400] == 10) > carthresh
            bigright = np.count_nonzero(currobs[i, :, 200:] == 10) > carthresh
            bigleft = np.count_nonzero(currobs[i, :, :600] == 10) > carthresh

            if abs(curract[i, 2]) <= 30:
                if right:
                    happening_info[2, i] = 1
                elif left:
                    happening_info[2, i] = 0
                else:
                    happening_info[2, i] = 2
            elif curract[i, 2] < -30:
                if bigleft:
                    happening_info[2, i] = 0
                elif right:
                    happening_info[2, i] = 1
                else:
                    happening_info[2, i] = 2
            elif curract[i, 2] > 30:
                if bigright:
                    happening_info[2, i] = 1
                elif left:
                    happening_info[2, i] = 0
                else:
                    happening_info[2, i] = 2

            print('frame '+str(i)+': '+movement[int(happening_info[0, i])] + ' and ' + pedestrians[int(happening_info[1, i])]
                  + ' and ' + vehicles[int(happening_info[2, i])])

        rle_happ = [[],[],[]]

        maxenframes = [40, 30, 24]

        for j in range(0, 3):
            enframes = maxenframes[j]
            for i in range(0, len(happening_info[j])):
                if ((j > 0) & (happening_info[0, i] < 2)) | (j < 2):
                    if i == 0:
                        enframes-=1
                    elif happening_info[j, i] == happening_info[j, i - 1]:
                        enframes-=1
                    else:
                        enframes = maxenframes[j]
                    if enframes <= 0:
                        if len(rle_happ[j]) == 0:
                            rle_happ[j].append(happening_info[j, i])
                        elif happening_info[j,i] != rle_happ[j][len(rle_happ[j])-1]:
                            rle_happ[j].append(happening_info[j, i])

        s1 = ''
        s2 = ''
        s3 = ''

        for k in range(0, min(3, len(rle_happ[0]))):
            if len(s1) == 0:
                s1 += movement[rle_happ[0][k]]
            else:
                s1 += ' and then ' + movement[rle_happ[0][k]]

        if np.count_nonzero(happening_info[1] == 0) >= 0.25*np.count_nonzero(happening_info[1] == 1):
            s2 = ' with other pedestrians around'

        for z in range(0, len(rle_happ[2])):
            if z == len(rle_happ[2])-1:
                if len(s3) > 0:
                    if (rle_happ[2][z] == 0) & (s3.__getitem__(10) != 'p'):
                        s3 = ' as a car approaches from the left'
                    elif (rle_happ[2][z] == 1) & (s3.__getitem__(10) != 'p') & (s3.__getitem__(31) != 'l'):
                        s3 = ' as a car approaches from the right'
                elif len(s3) == 0:
                    if rle_happ[2][z] == 0:
                        s3 = ' as a car approaches from the left'
                    elif rle_happ[2][z] == 1:
                        s3 = ' as a car approaches from the right'
            else:
                if (rle_happ[2][z] == 1) & (rle_happ[2][z+1] == 0):
                    s3 = ' as a car passes from right to left'
                elif (rle_happ[2][z] == 0) & (rle_happ[2][z+1] == 1):
                    s3 = ' as a car passes from left to right'
                elif len(s3) > 0:
                    if (rle_happ[2][z] == 0) & (s3.__getitem__(10) != 'p'):
                        s3 = ' as a car approaches from the left'
                    elif (rle_happ[2][z] == 1) & (s3.__getitem__(10) != 'p') & (s3.__getitem__(31) != 'l'):
                        s3 = ' as a car approaches from the right'
                elif len(s3) == 0:
                    if rle_happ[2][z] == 0:
                        s3 = ' as a car approaches from the left'
                    elif rle_happ[2][z] == 1:
                        s3 = ' as a car approaches from the right'

        print(rle_happ)

        final_sen = 'A pedestrian ' + s1 + s2 + s3

        print('For ' +filename[:len(filename)-4] + '_' + str(q+1) + ': ' + final_sen)

        json_string += final_sen + '"], "chCap": ["' + final_sen + '"]}, '

        print('###################################')

json_string = json_string[0:len(json_string)-2] + "]"

print(json_string)

with open('annotations.json', 'w') as outfile:
    outfile.write(json_string)

# b = dict(b)
#
# b['happening'] = happening_info
#
# np.savez(filename, **b)

# while True:
#     x = eval(input())
#     cv2.imshow('Frame', obs[x].reshape(450, 850))
#     print(movement[int(happening_info[x, 0])] + ' and ' + pedestrians[int(happening_info[x, 1])])
#
#     cv2.waitKey()
#     cv2.destroyAllWindows()


