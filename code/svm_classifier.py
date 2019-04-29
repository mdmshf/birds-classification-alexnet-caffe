
import numpy as np
import pickle as pkl
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.cross_validation import cross_val_score
import feature_extract as fe 


def run_alexnet():
    alexnet = fe.CaffeFeatureExtractor(
            model_path="../models/alexnet_deploy.prototxt",
            pretrained_path="../models/snapshots/caffe_alexnet_train_iter_10000.caffemodel",
            blob="fc7",
            crop_size=227,
            meanfile_path="../input/imagenet_mean.npy"
            )
    fe.create_dataset(net=alexnet, datalist="../input/train.txt", dbprefix="alexnet_train")
    fe.create_dataset(net=alexnet, datalist="../input/test.txt", dbprefix="alexnet_test")


def run(model_name):
    print "==> loading train data from %s" % (model_name + "_train_(features|labels).pkl")
    train_features = pkl.load(open(model_name + "_train_features.pkl"))
    train_labels = pkl.load(open(model_name + "_train_labels.pkl"))
    print "train_features.shape =", train_features.shape
    print "train_labels.shape =", train_labels.shape

    svm = LinearSVC(C=1.0)
    
    # print "==> training and test"
    # X_train = train_features[-1000:]
    # T_train = train_labels[-1000:]
    # X_test = train_features[:-1000]
    # T_test = train_labels[:-1000]
    # svm.fit(X_train, T_train)
    # Y_test = svm.predict(X_test)
    # print confusion_matrix(T_test, Y_test)
    # print accuracy_score(T_test, Y_test)
    # print classification_report(T_test, Y_test)
    
    
    print "==> cross validation"
    scores = cross_val_score(svm, train_features, train_labels, cv=10)
    print "Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std())

    
    svm.fit(train_features, train_labels)
    
    
    print "==> loading test data from %s" % (model_name + "_test_(features|labels).pkl")
    test_features = pkl.load(open(model_name + "_test_features.pkl"))
    
    
    print "==> predicting and writing"
    predicted_labels = svm.predict(test_features)
    with open("../input/test.txt") as fr:
        lines = fr.readlines()
    image_ids = []
    for line in lines:
        image_path = line.split()[0]
        image_name = line.split("/")[-1]
        image_id = image_name.split(".")[0]
        image_id = (image_id)
        image_ids.append(image_id)
    assert len(image_ids) == len(predicted_labels)
    test_ids=image_ids
    preds=predicted_labels
    with open(model_name+"_predict.txt","w") as f:
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
           # = "+str((right+wrong
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
                    #= "+str((right+wrong
        f.write("-------------\ncorrect = "+str(right)+"\nwrong = "+str(wrong)+"\ntotal="+str((right+wrong))+"\naccuracy = "+str((right*100)/(right+wrong))+"%\n----------------------")
        print("\n\nCONFUSION MATRIX\t\t\t\t|  ACCURACY\n")
        f.write("\n\nCONFUSION MATRIX\t\t\t\t|  ACCURACY\n\n")
        for i in range(len(arr)):
            for j in range(len(arr)):
                print(str(arr[i][j].astype(np.int))+"\t"),
                f.write(str(arr[i][j].astype(np.int))+"\t")
            print("|  %3.2f %%\n" % (arr[i][i]*100/arr[i].sum()))
            f.write("|  %3.2f %%\n\n" % (arr[i][i]*100/arr[i].sum()))
    f.close()




#    with open(model_name + "_predict.txt", "w") as fw:
    #fw.write("id,label\n")
     #for i in xrange(len(image_ids)):
      #      fw.write("%s,%d\n" % (image_ids[i], predicted_labels[i]))
        
if __name__ == "__main__":
    run_alexnet()
    run("alexnet")
