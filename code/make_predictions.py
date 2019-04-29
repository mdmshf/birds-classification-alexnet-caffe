
import sys
sys.path.append("/scratch/digits/deps/caffe/python")
import os
import glob
import cv2
import caffe
import lmdb
import numpy as np
from caffe.proto import caffe_pb2
#import seaborn as sns
import pandas as pd
#import matplotlibpyplot as plt
#sns.set(font_scale=2)
caffe.set_mode_gpu() 

#Size of images
IMAGE_WIDTH = 227
IMAGE_HEIGHT = 227

'''
Image processing helper function
'''

def transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

    #Image Resizing
    img = cv2.resize(img, (img_width, img_height), interpolation = cv2.INTER_CUBIC)

    return img


'''
Reading mean image, caffe model and its weights 
'''
#Read mean image
mean_blob = caffe_pb2.BlobProto()
with open('/scratch/digits/deps/caffe/examples/birds/input/imagenet_mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())
mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(
    (mean_blob.channels, mean_blob.height, mean_blob.width))


#Read model architecture and trained model's weights
net = caffe.Net('/scratch/digits/deps/caffe/examples/birds/models/alexnet_deploy.prototxt',
                '/scratch/digits/deps/caffe/examples/birds/models/snapshots/caffe_alexnet_train_iter_10000.caffemodel',
                caffe.TEST)

#Define image transformers
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

'''
Making predicitions
'''
#Reading image paths
test_img_paths = [img_path for img_path in glob.glob("../input/test1/*jpg")]

#Making predictions
test_ids = []
preds = []
for img_path in test_img_paths:
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = transform_img(img, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT)
    
    net.blobs['data'].data[...] = transformer.preprocess('data', img)
    out = net.forward()
    pred_probas = out['prob']

    test_ids = test_ids + [img_path.split('/')[-1][:-4]]
    preds = preds + [pred_probas.argmax()]

    print img_path
    print pred_probas.argmax()
    print '-------'

'''
Making submission file
'''
with open("../test_model.csv","w") as f:
    f.write("id,label,\n")
    right=0
    wrong=0
    y_actu=[]
    y_pred=[]
    arr = np.zeros((6,6),dtype=int)
    for i in range(len(test_ids)):
        y_pred.append(preds[i])
        if 'egr' in str(test_ids[i]):
            y_actu.append(0)
            if (preds[i] == 0):
                f.write(str(test_ids[i])+","+str(preds[i])+",correct\n")
                right+=1
                arr[0][0]+=1
            else:
                f.write(str(test_ids[i])+","+str(preds[i])+",wrong\n")
                wrong+=1
                arr[0][preds[i]]+=1
        if 'man' in str(test_ids[i]):
            y_actu.append(1)
            if (preds[i] == 1):
                f.write(str(test_ids[i])+","+str(preds[i])+",correct\n")
                right+=1
                arr[1][1]+=1
            else:
                f.write(str(test_ids[i])+","+str(preds[i])+",wrong\n")
                wrong+=1
                arr[1][preds[i]]+=1
        if 'owl' in str(test_ids[i]):
            y_actu.append(2)
            if (preds[i] == 2):
                f.write(str(test_ids[i])+","+str(preds[i])+",correct\n")
                right+=1
                arr[2][2]+=1
            else:
                f.write(str(test_ids[i])+","+str(preds[i])+",wrong\n")
                wrong+=1
                arr[2][preds[i]]+=1
        if 'puf' in str(test_ids[i]):
            y_actu.append(3)
            if (preds[i] == 3):
                f.write(str(test_ids[i])+","+str(preds[i])+",correct\n")
                right+=1
                arr[3][3]+=1
            else:
                f.write(str(test_ids[i])+","+str(preds[i])+",wrong\n")
                wrong+=1
                arr[3][preds[i]]+=1
        if 'tou' in str(test_ids[i]):
            y_actu.append(4)
            if (preds[i] == 4):
                f.write(str(test_ids[i])+","+str(preds[i])+",correct\n")
                right+=1
                arr[4][4]+=1
            else:
                f.write(str(test_ids[i])+","+str(preds[i])+",wrong\n")
                wrong+=1
                arr[4][preds[i]]+=1
        if 'wod' in str(test_ids[i]):
            y_actu.append(5)
            if (preds[i] == 5):
                f.write(str(test_ids[i])+","+str(preds[i])+",correct\n")
                right+=1
                arr[5][5]+=1
            else:
                f.write(str(test_ids[i])+","+str(preds[i])+",wrong\n")
                wrong+=1
                arr[5][preds[i]]+=1
    f.write("-------------\ncorrect = "+str(right)+"\nwrong = "+str(wrong)+"\ntotal = "+str((right+wrong))+"\naccuracy = "+str((right*100)/(right+wrong))+"%\n----------------------")
    print("\n\nCONFUSION MATRIX\t\t\t\t|  ACCURACY\n")
    f.write("\n\nCONFUSION MATRIX\t\t\t\t|  ACCURACY\n\n")
    for i in range(len(arr)):
        for j in range(len(arr)):
            print(str(arr[i][j].astype(np.int))+"\t"),
            f.write(str(arr[i][j].astype(np.int))+"\t")
        print("|  %3.2f %%\n" % (arr[i][i]*100/arr[i].sum()))
        f.write("|  %3.2f %%\n\n" % (arr[i][i]*100/arr[i].sum()))
f.close()
