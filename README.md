# OCFER

预训练模型的数据增强和超参数设置是参考的是这个仓库：https://github.com/LetheSec/Fer2013-Facial-Emotion-Recognition-Pytorch 
在此非常感谢
这个仓库所包含的内容是One-Class Classification for Facial Expression Recognition
## data
**FER2013**：https://www.kaggle.com/datasets/deadskull7/fer2013
**AffectNet**：https://www.kaggle.com/datasets/arafatshovon/affectnet
下载这两个数据集放在<span style="background-color: #444; padding: 3px 5px;">data/</span>下

## 预训练模型
**Fer2013**:<span style="background-color: #111; padding: 3px 5px;">checkpoint/
              Fer2013/
              best_checkpoint.tar</span>

**AffectNet**: <span style="background-color: #111; padding: 3px 5px;">checkpoint/
                 AffectNet/
                  best_checkpoint.tar</span>
## train
注意训练单类模型请修改<span style="background-color: #111; padding: 3px 5px;">paramers.py</span>文件中的target_class
python train_fer2013.py
