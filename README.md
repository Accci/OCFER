# OCFER

The pre-trained model's data augmentation and hyperparameter settings were referenced from this repository: [https://github.com/LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch](https://github.com/LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch). We greatly appreciate it.<br>
Please refer to the `PreTrain` folder for more details.<br>

This repository contains One-Class Classification for Facial Expression Recognition.<br>

Data<br>
## data
**FER2013**：[https://www.kaggle.com/datasets/deadskull7/fer2013](https://www.kaggle.com/datasets/deadskull7/fer2013])<br>
**AffectNet**：[https://www.kaggle.com/datasets/arafatshovon/affectnet](https://www.kaggle.com/datasets/arafatshovon/affectnet)<br>
Download these two datasets and place them in the`data/`directory.<br>

## 预训练模型
**Fer2013**:`checkpoint/Fer2013/best_checkpoint.tar`<br>

**AffectNet**: `checkpoint/AffectNet/best_checkpoint.tar`<br>
## train
To train a single-class model, please modify the `paramers.py`in the`target_class`file.<br>
```python train_fer2013.py```<br>
For AffectNet, note that the `target_class` is the folder name; directly modify it in the `train_affect.py` script:<br>
```python train_affect.py```<br>
