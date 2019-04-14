
#the director where sotre the weight, the prediction will be stored in the same folder
dir=/media/lq/C13E-1ED0/dataset/UCF_Crimes/results/adam_c3d_final_1_pre_train_1M/weights/
weight=weights.01-0.597.hdf5

#the name of the weight (*.hdf5)s
#the path of the videos
video_dir=/media/lq/C13E-1ED0/dataset/UCF_Crimes/Videos/RoadAccidents_test/

python -m predict.ucf_predict -d=$dir -w=$weight -v=$video_dir 