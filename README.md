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

2. run
```
python run.py --bert_model=bert-base-uncased
--output_dir=./outupt
--data_dir=./data/twitter2015 or 2017
--task_name=twitter2015 or 2017
--do_train
```
3. test
```
python test.py --bert_model=roberta-large-uncased
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
 

# Acknowledgements

<!---Our framework and some codes are based on [HireMLP](https://github.com/liuruiyang98/Jittor-MLP), thanks very much!-->

[1] Sun H, Wang H, Liu J, et al. CubeMLP: An MLP-based model for multimodal sentiment analysis and depression estimation[C]//Proceedings of the 30th ACM international conference on multimedia. 2022: 3722-3729.
[2] Guo J, Tang Y, Han K, et al. Hire-mlp: Vision mlp via hierarchical rearrangement[C]//Proceedings of the ieee/cvf conference on computer vision and pattern recognition. 2022: 826-836.

