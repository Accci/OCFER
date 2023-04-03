from utils import *
from parameters import *

import numpy as np
# from data import *
from affect_dataloader import *
haper_para = hyperparameters()
def main():
    eval(hyper_para)
    
def eval(hyper_para):
    model = get_model(hyper_para.arch)
    clf_list = []

    clf0 =load_model('AffectNet_class1_model.pkl')
    clf_list.append(clf0)
    clf1 =load_model('AffectNet_class2_model.pkl')
    clf_list.append(clf1)
    clf2 =load_model('AffectNet_class3_model.pkl')
    clf_list.append(clf2)
    clf3 =load_model('AffectNet_class4_model.pkl')
    clf_list.append(clf3)
    clf4 =load_model('AffectNet_class5_model.pkl')
    clf_list.append(clf4)
    clf5 =load_model('AffectNet_class6_model.pkl')
    clf_list.append(clf5)
    clf6 =load_model('AffectNet_class7_model.pkl')
    clf_list.append(clf6)
    clf7 =load_model('AffectNet_class8_model.pkl')
    clf_list.append(clf6)
    
    test_features = []
    test_labels = []
    _,test_loader= get_dataloaders(bs=hyper_para.bs)
    for i, data in  enumerate(test_loader):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()
        
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
    print("Test_Accuracy:", accuracy)
    

if __name__ == '__main__':
    main()
