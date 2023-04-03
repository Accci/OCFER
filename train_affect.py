import sys
import argparse

from utils import *
from parameters import *

import numpy as np
from affect_data import *

from sklearn.svm import OneClassSVM

import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
    
haper_para = hyperparameters()
best_acc = 0
def main():
    global best_acc
    for i in range(haper_para.epoch):
        train_loader,test_loader= get_dataloaders(path='/data/affectnet',bs = hyper_para.bs,target_class='class001')
        model = get_model(hyper_para.arch)
        if(hyper_para.gpu_flag):
            model.cuda()
        val_acc, clf = train_oc_svm(train_loader,test_loader,model)
        if val_acc > best_acc:
            best_acc = max(val_acc, best_acc)
            print(f"save the best model: {best_acc}")
            with open('AffectNet_class1_model.pkl', 'wb') as f:
                pickle.dump(clf, f)
        
   
#         eval(haper_para)

def train_oc_svm(train_loader,test_loader,model): 
    train_features = []
    train_labels = []
    correct = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.cuda()
        labels = labels.cuda()
         
        bs, ncrops, c, h, w = inputs.shape
        inputs = inputs.view(-1, c, h, w)
        labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)

        features = model(inputs)
        features = features.reshape(features.size(0), -1)
        train_features.append(features.data.cpu().numpy())
        train_labels.append(labels.data.cpu().numpy())
        
    train_features = np.concatenate(train_features, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    clf = OneClassSVM()
    param_grid = {'nu': [0.05,0.1,0.15,0.2],
              'kernel': ['rbf', 'sigmoid','poly']}
    scorer = make_scorer(f1_score)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorer)
    grid_search.fit(train_features,train_labels)
    print("Best parameter combination: ", grid_search.best_params_)
    print("Best F1 score: ", grid_search.best_score_)
    clf = OneClassSVM(nu=grid_search.best_params_['nu'],
                    kernel=grid_search.best_params_['kernel'])
#     clf = OneClassSVM(kernel='sigmoid', nu = 0.07)
    clf.fit(train_features)
    
    test_features = []
    test_labels = []
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
    
    preds = clf.predict(test_features)
    test = test_labels.flatten() 
    val_acc = accuracy_score(test, preds)
    print("accuracy_score1: {:.2f}%".format(accuracy_score(test, preds)*100))
    print("precision_score: {:.2f}%".format(precision_score(test, preds)*100))
    print("recall_score: {:.2f}%".format(recall_score(test,preds)*100))
    print("f1_score: {:.2f}%".format(f1_score(test,preds)*100))
    
    return val_acc, clf
    

if __name__ == '__main__':
    main()
