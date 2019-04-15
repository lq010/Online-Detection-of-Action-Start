All the code available is to reproduce [ODAS][1]. It will be explained step by step all the stages required to reproduce the results on THUMOS'14 and also how to obtain predictions with the model.

# Reproduce Experiments

## Download the pre-trained weights
Check if the `weight/sports1M_weights_tf.h5` is correctly downloaded, if not, you can: (1) install git-lfs, run `git lfs pull`; (2) download it [here][2].

## Download the THUMOS'14 dataset
In this project, I use validation videos (`THUMOS14/data/download_links_val.txt`) for training, test videos(`THUMOS14/data/download_links_test.txt`) for testing.

## Pre-processing
- Convert validation videos to images:
    1. Go into the `THUMOS14/data` dir, set the video paths (`videoPaths.py`) 
    2. Run `python video2image.py`

- Randomly split validation videos into train-set and val-set:

    Note that the validation videos has been split into train-set (train.json) and val-set (validation.json). If you want to re-split the data, run `python gen_train_val_sets.py`

## Train
```badh
>>$cd THUMOS14
```
- Adaptive sampling
```bash
>>$python train_c3d.py  -h
Using TensorFlow backend.
usage: train_c3d.py [-h] [-id EXPERIMENT_ID]

Train the c3d model (adaptive sampling)

optional arguments:
  -h, --help         show this help message and exit
  -id EXPERIMENT_ID  Experiment ID to track and not overwrite resulting models 
```
- model the temporal consistency
```bash

>>$python train_c3d_TC.py  -h
Using TensorFlow backend.
usage: train_c3d_TC.py [-h] [-id EXPERIMENT_ID]

Train the c3d model (adaptive sampling + temporal consistency)

optional arguments:
  -h, --help         show this help message and exit
  -id EXPERIMENT_ID  Experiment ID to track and not overwrite resulting models
```
-Generate hard-negative via GAN
```bash
>>$python train_c3d_TC_GAN.py  -h
Using TensorFlow backend.
usage: train_c3d_TC_GAN.py [-h] [-id EXPERIMENT_ID] [-w PRETRAINED_WEIGHTS]

Train the GAN model

optional arguments:
  -h, --help            show this help message and exit
  -id EXPERIMENT_ID     Experiment ID to track and not overwrite resulting
                        models. (default: test)
  -w PRETRAINED_WEIGHTS
                        The pretrained weights, the weights will be used to
                        initialize the GAN model. (default: results/adam_temp
                        oral_8/weights/weights.02-2.111.hdf5)
```

## Predict
1. Open the file `predict.sh`, set the director (`dir`) and name (`weight`) of the weights file.
2. run `./predict.sh`. A new file, `prediction_*.hdf5`, will be created in the the folder `dir`.
3. Open the file `create_file_prediction_file.sh`, set the director (`dir`) and name (`rawPredictionFile`) of the `predictions_*.hdf5` files.
4. run `./create_final_pediction_file.sh`. A json file, `predictions_*.json` will be created.

## Evaluate 

1. Open file `evaluate.sh`, set the path of the `predictions_*.json` file.
2. run `./evaluate.sh`.


[1]:http://openaccess.thecvf.com/content_ECCV_2018/html/Zheng_Shou_Online_Detection_of_ECCV_2018_paper.html

[2]: https://drive.google.com/file/d/1vkSw6yKe4CCq4SYBPmoTofVlXj-MBYms/view?usp=sharing