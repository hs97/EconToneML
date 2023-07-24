# EconToneML: Convolutional Neural Networks for Tone, Gender, and Age Imputation

This repository hosts the trained models in [Handlan and Sheng (2023)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4316513) as well as a Jupyter notebook demo of how to apply the trained models to classifying audio clips.

The demo notebook can be found under the `demo` folder. The demo uses two audio clips from the NBER 2023 Summer Institute Methods Lectures. These lectures are publicly available and downloadable on Youtube. The downloaded audio clips are stored in `data/NBER`.

The trained models are under the `model` folder. The models are trained with 5-fold cross validation for each gender and label. There are a total of 25 models.

If you want to use our models, we would appreciate you citing our paper as the following:

> Handlan, Amy and Sheng, Haoyu, Gender and Tone in Recorded Economics Presentations: Audio Analysis with Machine Learning (January 1, 2023). Available at SSRN: https://ssrn.com/abstract=4316513 or http://dx.doi.org/10.2139/ssrn.4316513