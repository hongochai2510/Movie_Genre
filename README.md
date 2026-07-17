# Multi-label Movie Genre Classification from Posters
## Introduction

Movie posters often contain visual information that reflects the genre, mood, and theme of a film. This project focuses on the task of **multi-label movie genre classification** using only poster images.
The proposed system combines deep visual features extracted by **DenseNet-169** with object-level semantic features detected by **YOLOv8**. These features are fused and used to train machine learning classifiers such as **LightGBM** and **XGBoost** for predicting multiple movie genres simultaneously.

![image alt](https://github.com/hongochai2510/Movie_Genre/blob/128878c4468d53b1639069a088cfb2d97b4d0e5b/bt.png)

## Objectives

- Build a multi-label movie genre classification system.
- Extract visual representations using DenseNet-169.
- Detect semantic objects using YOLOv8.
- Reduce feature dimensionality using Truncated SVD, Feature Selection and Non-SVD.
- Compare LightGBM and XGBoost.
- Optimize prediction thresholds for each genre.

## Pipeline
![image alt](https://github.com/hongochai2510/Movie_Genre/blob/6d0fe8049bd4e8d8cf3abd96a2c45cc845e897a2/pipe.png)
