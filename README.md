# RR-Mixer A Rearrangement and Restore Mixer Model for Target-Oriented Multimodal Sentiment Classification
Initial version codes for RR-Mixer: RR-Mixer A Rearrangement and Restore Mixer Model for Target-Oriented Multimodal Sentiment Classification
 

![model](https://github.com/LiLi-Jia/RR-Mixer/assets/44886362/38b5be68-1eda-49fc-b6b8-d3cef22be182)

## Requirement

- Python 3.7
- NVIDIA GPU + CUDA cuDNN
- PyTorch 1.9.0

# Data

1. The image-text data public datasets used in this paper are [TWITTER-15](https://github.com/jefferyYu/TomBERT) and [TWITTER-17](https://github.com/jefferyYu/TomBERT).
2. Train a visual sentiment classification model based on the ResNet-152. This datasets is provided by [Yang J[1]](http://47.105.62.179:8081/sentiment_web/datasets/LDL.tar.gz).
3. The Object Score and IoU Score in the image are obtained using Yolov5. Also, the Senti_score is obtained using the pre-trained model from step 2.

## Run
1. search and replace relevant paths
   res_path = 'feature path'

2. train the model
```
python train_and_test.py --bert_model=bert-base-uncased
--output_dir=./outupt
--data_dir=./data/twitter2015 or 2017
--task_name=twitter2015 or 2017
--do_train
```
3. test the model
```
python train_and_test.py --bert_model=bert-base-uncased
--output_dir=./outupt
--data_dir=./data/twitter2015 or 2017
--task_name=twitter2015 or 2017
--do_eval
```

## Citation
If you find this useful for your research, please use the following.

```
@ARTICLE{10354512,
  author={Jia, Li and Ma, Tinghuai and Rong, Huan and Sheng, Victor S. and Huang, Xuejian and Xie, Xintong},
  journal={IEEE Transactions on Artificial Intelligence}, 
  title={A Rearrangement and Restore Mixer Model for Target-Oriented Multimodal Sentiment Classification}, 
  year={2023},
  volume={},
  number={},
  pages={1-11},
  keywords={Task analysis;Transformers;Image restoration;Mixers;Visualization;Artificial intelligence;Feature extraction;Feature Mixing;rearrangement and restore operations;MLPs-based;target-oriented multimodal sentiment classification},
  doi={10.1109/TAI.2023.3341879}}
```
```
@INPROCEEDINGS{9880243,
  author={Guo, Jianyuan and Tang, Yehui and Han, Kai and Chen, Xinghao and Wu, Han and Xu, Chao and Xu, Chang and Wang, Yunhe},
  booktitle={2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={Hire-MLP: Vision MLP via Hierarchical Rearrangement}, 
  year={2022},
  volume={},
  number={},
  pages={816-826},
  keywords={Representation learning;Computer vision;Image segmentation;Semantics;Computer architecture;Object detection;Transformers;Deep learning architectures and techniques; Efficient learning and inferences; Machine learning; Recognition: detection;categorization;retrieval; Representation learning},
  doi={10.1109/CVPR52688.2022.00090}}
```

# Acknowledgements
<!---Our framework and some codes are based on [HireMLP](https://github.com/liuruiyang98/Jittor-MLP), thanks very much!-->
[1] Yang J, Sun M, Sun X. Learning visual sentiment distributions via augmented conditional probability neural network[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2017, 31(1).
