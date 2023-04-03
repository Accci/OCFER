import logging
import os
import random
import csv
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

from models import *
from data import *

import cv2
import scipy.io
import json
from scipy import misc
from sklearn import svm
from PIL import Image
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import glob
import os

from sklearn.svm import OneClassSVM

import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer



class classifier_nn(nn.Module):

	def __init__(self,D):
            super(classifier_nn,self).__init__()
            self.fc1 = nn.Linear(D, 2)
		
	def forward(self,x):
            out = x
            out = self.fc1(out)
            return out

# class classifier_nn(nn.Module):

# 	def __init__(self,D):
#             super(classifier_nn,self).__init__()
#             self.classifier = nn.Sequential(nn.Linear(D, 256),
#                                         nn.ReLU(inplace=True),
#                                         nn.Linear(256, 128),
#                                         nn.ReLU(inplace=True),
#                                         nn.Linear(128, 64),
#                                         nn.ReLU(inplace=True),
#                                         nn.Linear(64, 2))
# 	def forward(self,x):
#             out = x
#             out = self.classifier(out)
#             return out
        
def AddNoise(inputs, sigma):

	noise_shape = np.shape(inputs)
	
	noise = np.random.normal(0, sigma, noise_shape)
	noise = torch.from_numpy(noise)
	noise = torch.autograd.Variable(noise).float()

	if(inputs.is_cuda):
		outputs = inputs + noise.cuda()
	else:
		outputs = inputs + noise

	return outputs

def get_model(arch='ResNet18'):
    if arch == 'ResNet18':
        model = ResNet18()
        model = model.cuda()
        model.load_state_dict(torch.load('./res.tar')['model_state_dict'])
        for name, param in model.named_parameters():
            if 'conv' in name:
                param.requires_grad = False
        new_model=nn.Sequential(*list(model.children())[:-1])
    elif arch == 'CBAM':
        model = CBAM_ResNet18()
        model = model.cuda()
        model.load_state_dict(torch.load('./checkpoint/Fer2013/best_checkpoint.tar')['model_state_dict']) 
        for name, param in model.named_parameters():
            if 'conv' in name:
                param.requires_grad = False
        new_model=nn.Sequential(*list(model.children())[:-1])
        
    return new_model


class ConfusionMatrix(object):
    """
    注意，如果显示的图像不全，是matplotlib版本问题
    本例程使用matplotlib-3.2.1(windows and ubuntu)绘制正常
    需要额外安装prettytable库
    """

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()
        
def CM(num_classes=2):
    json_label_path = './class_indices.json'
    assert os.path.exists(json_label_path), "cannot find {} file".format(json_label_path)
    json_file = open(json_label_path, 'r')
    class_indict = json.load(json_file)

    labels = [label for _, label in class_indict.items()]
    confusion = ConfusionMatrix(num_classes=num_classes, labels=labels)

    return confusion



def oc_cnn(class_number, hyper_para, epoch):
    classifier = classifier_nn(hyper_para.D)
              
#     train_loader, val_loader, test_loader= get_dataloaders()
    train_loader, val_loader = get_dataloaders()
    model = get_model(hyper_para.arch)
#     cm = CM(num_classes=2)
    if(hyper_para.gpu_flag):
        model.cuda()
        classifier.cuda()
	
	# loss functions
    criterion = nn.CrossEntropyLoss()
    
    criterion.cuda()

    model_optimizer = torch.optim.SGD( model[-3:].parameters(), lr=hyper_para.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
#     model_optimizer = torch.optim.SGD(model.parameters(), lr=hyper_para.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    
    classifier_optimizer = torch.optim.SGD(classifier.parameters(), lr=hyper_para.lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            classifier_optimizer, T_max=20)
    
    train_loss, train_acc = train(train_loader, model, classifier,criterion,model_optimizer,classifier_optimizer,epoch,hyper_para)
    val_loss, val_acc = evaluate(model,classifier, val_loader)
#     test_loss, test_acc = evaluate(model,classifier, test_loader,cm)
    print("epoch:{0}\t train_loss:{1:.4f}\t train_acc:{2:.4f}%\t val_loss:{3:.4f} \t val_acc:{4:.4f}%\t ".format(epoch,train_loss, train_acc*100, val_loss,val_acc*100))
              
def train(train_loader, model, classifier, criterion, model_optimizer, classifier_optimizer, epoch, hyper_para):
    classifier.train()          
    model.train()
    count = 0
    correct = 0
    train_loss = 0

    for i, (inputs, labels) in enumerate(train_loader):

        inputs = inputs.cuda()
        labels = labels.cuda()
         
        bs, ncrops, c, h, w = inputs.shape
        inputs = inputs.view(-1, c, h, w)
        labels = torch.repeat_interleave(labels, repeats=ncrops, dim=0)
            
        labels = np.concatenate( (np.zeros((bs*ncrops,)), (np.ones(bs*ncrops,)) ), axis=0)
        labels = torch.from_numpy(labels)
        labels = torch.autograd.Variable(labels.cuda()).long()
      
         
#         gaussian_data = np.random.normal(0, hyper_para.sigma, (bs*ncrops, hyper_para.D))
        gaussian_data = np.random.normal(0, hyper_para.sigma, (bs*ncrops, hyper_para.D))
        gaussian_data = torch.from_numpy(gaussian_data)
        # compute output
        out1 = model(AddNoise(inputs, hyper_para.sigma1))
        avg = nn.AdaptiveAvgPool2d(1)
        out1 = avg(out1)
        out1 = out1.view(bs*ncrops, -1)
        out2 = torch.autograd.Variable(gaussian_data.cuda()).float()
        out = torch.cat( (out1, out2),0)
        out = F.relu(out)
        outputs  = classifier(out)
               
        loss = criterion(outputs, labels)
        loss.backward()
      
        model_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        
        # compute gradient and do SGD step

        model_optimizer.step()
        classifier_optimizer.step()

        train_loss +=loss
        _, preds = torch.max(outputs, dim= 1)
        correct += torch.sum(preds == labels.data).item()
        count += labels.shape[0]
    return train_loss / count, correct / count

def evaluate(model,classifier, val_loader):
    model.eval()
    classifier.eval()
    correct = 0
    count = 0
    val_loss = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            if True:
                # fuse crops and batchsize
                bs, ncrops, c, h, w = images.shape
                images = images.view(-1, c, h, w)

                # forward
                outputs = model(images)
                avg = nn.AdaptiveAvgPool2d(1)
                outputs = avg(outputs)
                outputs = outputs.view(bs*ncrops,-1)
                outputs = classifier(outputs)
                # combine results across the crops
                outputs = outputs.view(bs,ncrops, -1)
                outputs = torch.sum(outputs, dim=1) / ncrops

            else:
                outputs = model(images)

            loss = nn.CrossEntropyLoss()(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, dim= 1)
            correct += torch.sum(preds == labels.data).item()
            count += labels.shape[0]
#             cm.update(preds.to("cpu").numpy(), labels.to("cpu").numpy())
#     cm.plot()
#     cm.summary()

    return val_loss / count, correct / count


def load_model(file):
    with open(file, 'rb') as f:
        clf = pickle.load(f)
    return clf;
