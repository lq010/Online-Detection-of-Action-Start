# Thesis
This is a project trying to implement [ODAS][1].

## Requirements:
- python3
- keras 2.2.0
- tensorflow 1.6.0
- OpenCV 3.3.1
- tqdm

## Usage
#### THUMOS'14 dataset:

- Train

|   training method  | command | initialization(weights) |
|:-----------:|:---------------:|:----------:|
|   adaptive     | `python train_c3d.py`|  sports-1M   |
| adaptive+TC  | `python train_c3d_TC.py` |   sports-1M   |
| adaptive+TC+GAN  | `python train_c3d_TC_GAN.py`  |   adaptive+TC    |       
    
- Predict:
1. `./predict.sh` -> outputs predictions of input windows 
2. `./create_final_prediction_file.sh` -> outputs a json file, which contains the AS predictions

-Evaluate:
1. `./evaluation.sh`

#### UCF(roadAccidents) dataset:

- Train

|   training method  | command | initialization(weights) |
|:-----------:|:---------------:|:----------:|
| pre-train (A+BG windows)  | `python train_c3d_pre_train.py`  |   sports-1M   | 
|   adaptive     | `python train_c3d_re_train.py`|  pre-train  |
| adaptive+TC  | `python train_c3d_TC.py` |   pre-train   |
| adaptive+TC+GAN  | `python train_c3d_TC_GAN.py`  |   adaptive+TC    |       
    
- Predict:
1. `./predict.sh` -> outputs predictions of input windows 
2. `./create_final_prediction_file.sh` -> outputs a json file, which contains the AS predictions

-Evaluate:
1. `./evaluation.sh`

[1]:http://openaccess.thecvf.com/content_ECCV_2018/html/Zheng_Shou_Online_Detection_of_ECCV_2018_paper.html