# CheXNet

This is part of my internship at Endimension Technology, IIT Bombay.

PyTorch implementation of [CheXNet: Radiologist level pneumonia detection using deep learning](https://arxiv.org/abs/1711.05225)
based on [this](https://github.com/arnoweng/CheXNet) implementation.

You can run the complete notebook on Kaggle -> [CheXNet-PyTorch](https://www.kaggle.com/abhiswain/chexnet-pytorch)

## Hyperparameters

Batch size | Learning Rate | Epochs | Time
-----------|---------------|--------|------
64 | 0.01 | 20 | 2 hrs

## Per Class AUROC

Pathology | AUROC
----------|-------
Atelectasis | 0.735
Cardiomegaly | 0.882
Effusion | 0.82
Infiltration | 0.673
Mass | 0.788
Nodule | 0.728
Pneumonia | 0.647
Pneumothorax | 0.799
Consolidation | 0.689
Edema | 0.832
Emphysema | 0.858
Fibrosis | 0.77
Pleural_Thickening | 0.719
Hernia | 0.846



### You can find the internship certificate [here](https://github.com/Abhiswain97/CheXNet/blob/master/Internship%20Certificate-1.png)
