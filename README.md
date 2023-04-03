# OCFER

预训练模型的数据增强和超参数设置是参考的是这个仓库：https://github.com/LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch <br>
在此非常感谢<br>
***
这个仓库所包含的内容是One-Class Classification for Facial Expression Recognition<br>
## data
**FER2013**：https://www.kaggle.com/datasets/deadskull7/fer2013<br>
**AffectNet**：https://www.kaggle.com/datasets/arafatshovon/affectnet<br>
下载这两个数据集放在`data/`
下<br>

## 预训练模型
**Fer2013**:`checkpoint/
              Fer2013/
              best_checkpoint.tar`<br>

**AffectNet**: `checkpoint/
                 AffectNet/
                  best_checkpoint.tar`<br>
## train
注意训练单类模型请修改`paramers.py`文件中的`target_class`<br>
`python train_fer2013.py`
注意`AffectNet`中的`targetz_class`是文件夹名称，直接在`train_affect.py`中修改
`python train_affect.py`
