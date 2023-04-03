import itertools
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from models import *
import torch.nn.functional as F
# from dataset import get_dataloaders
from affectnet import get_dataloaders

# class_names = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprised", "Neutral"]
class_names = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger","Contempt"]
checkpoint_name = 'CBAM'
def main():
    model = CBAM_ResNet18()
    model.load_state_dict(torch.load('/results/affect/affect1/CBAM_epoch200_bs32_lr0.1_momentum0.9_wd0.0001_seed0_smoothTrue_mixupTrue_schedulerreduce_affect1/checkpoints/best_checkpoint.tar')['model_state_dict'])
    model.cuda()
    model.eval()
    _,val_loader=get_dataloaders()
    count = 0
    correct = 0
    all_target = []
    all_output = []
    with torch.no_grad():
        for i,data in enumerate(val_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # fuse crops and batchsize
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w)

            # forward
            outputs = model(inputs)

            # combine results across the crops
            outputs = outputs.view(bs, ncrops, -1)
            outputs = torch.sum(outputs, dim=1) / ncrops
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data).item()
            count += labels.shape[0]
            
            all_target.append(labels.data.cpu())
            all_output.append(preds.data.cpu())
            
    all_target = np.concatenate(all_target)
    all_output = np.concatenate(all_output)

    matrix = confusion_matrix(all_target, all_output)
    np.set_printoptions(precision=2)

    plot_confusion_matrix(
        matrix,
        classes=class_names,
        normalize=True,
        # title='{} \n Accuracc: {:.03f}'.format(checkpoint_name, acc)
        title="Resnet18 With CBAM",
    )

    # plt.show()
    # plt.savefig('cm_{}.png'.format(checkpoint_name))
    plt.savefig("./cm_{}.pdf".format(checkpoint_name))
    plt.close()
   
    
        


# 生成混淆矩阵


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")
    print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title, fontsize=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label", fontsize=12)
    plt.xlabel("Predicted label", fontsize=12)
    plt.tight_layout()

if __name__ == "__main__":
    main()
