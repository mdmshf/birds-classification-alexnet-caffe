import os
import sys
sys.path.append("/scratch/digits/deps/caffe/python")
import numpy as np
import caffe

if __name__ == "__main__":
    data = open("../input/imagenet_mean.binaryproto", "rb").read()
    blob = caffe.proto.caffe_pb2.BlobProto()
    blob.ParseFromString(data)

    arr = np.asarray(caffe.io.blobproto_to_array(blob))
    print arr.shape

    np.save("../input/imagenet_mean.npy", arr)
