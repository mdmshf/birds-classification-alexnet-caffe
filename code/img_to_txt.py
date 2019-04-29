import glob

test_img_paths = [img_path for img_path in glob.glob("/scratch/digits/deps/caffe/examples/birds/input/test1/*jpg")]
train_img_paths = [img_path for img_path in glob.glob("/scratch/digits/deps/caffe/examples/birds/input/train/*jpg")]
aw=['egr','man','wod','owl','puf','tou']

with open("../input/test.txt","w") as f:
    for i in test_img_paths:
        for j in range(len(aw)):
            if aw[j] in i:
                label=j
        f.write(i+" "+str(label)+"\n")
f.close()

with open("../input/train.txt","w") as f:
    for i in train_img_paths:
        for j in range(len(aw)):
            if aw[j] in i:
                label=j
        f.write(i+" "+str(label)+"\n")
f.close()
