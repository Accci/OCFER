from utils import *
from parameters import *

import numpy as np
# from data import *
from fer_dataloader import *
haper_para = hyperparameters()
def main():
    eval(hyper_para)
    
def eval(hyper_para):
    model = get_model(hyper_para.arch)
    clf_list = []
    val_features = []
    val_labels = []
    test_features = []
    test_labels = []
    clf0 =load_model('one_class_svm_anger_model.pkl')
    clf_list.append(clf0)
    clf1 =load_model('one_class_svm_disgust_model.pkl')
    clf_list.append(clf1)
    clf2 =load_model('one_class_svm_fear_model.pkl')
    clf_list.append(clf2)
    clf3 =load_model('one_class_svm_happy_model.pkl')
    clf_list.append(clf3)
    clf4 =load_model('one_class_svm_sad_model.pkl')
    clf_list.append(clf4)
    clf5 =load_model('one_class_svm_surprised_model.pkl')
    clf_list.append(clf5)
    clf6 =load_model('one_class_svm_normal_model.pkl')
    clf_list.append(clf6)
    
    val_loader, test_loader= get_dataloaders()
    for i, data in  enumerate(val_loader):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        if True:
          # fuse crops and batchsize
            bs, ncrops, c, h, w = images.shape
            images = images.view(-1, c, h, w)

        features = model(images)
        features = features.reshape(bs,ncrops , -1)
        features = torch.sum(features, dim=1) / ncrops
        val_features.append(features.data.cpu().numpy())
        val_labels.append(labels.data.cpu().numpy())
        
    val_features = np.concatenate(val_features, axis=0)
    val_labels = np.concatenate(val_labels, axis=0)
    val_labels = val_labels.flatten()
    
    val_pred = []
    for val in val_features:
        scores = [clf.decision_function(val.reshape(1,-1)) for clf in clf_list]
        val_pred.append(np.argmax(scores))
#     print(val_pred)
    accuracy = accuracy_score(val_labels, val_pred)
    print("PrivateTest_Accuracy:", accuracy)
    
    for i, data in  enumerate(test_loader):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        if True:
          # fuse crops and batchsize
            bs, ncrops, c, h, w = images.shape
            images = images.view(-1, c, h, w)

        features = model(images)
        features = features.reshape(bs,ncrops , -1)
        features = torch.sum(features, dim=1) / ncrops
        test_features.append(features.data.cpu().numpy())
        test_labels.append(labels.data.cpu().numpy())
        
    test_features = np.concatenate(test_features, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    test_labels = test_labels.flatten()
    
    test_pred = []
    for test in test_features:
        scores = [clf.decision_function(test.reshape(1,-1)) for clf in clf_list]
        test_pred.append(np.argmax(scores))
#     print(val_pred)
    accuracy = accuracy_score(test_labels, test_pred)
    print("PubilcTest_Accuracy:", accuracy)


if __name__ == '__main__':
    main()
