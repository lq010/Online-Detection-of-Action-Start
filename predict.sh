
#the director where sotre the weight, the prediction will be stored in the same folder
dir=/home/lq/Documents/Thesis/Thesis/results/adam_temporal_6/best_weight/
#the name of the weight (*.hdf5)
weight=weights.02-2.20.hdf5
#the path of the videos
video_dir=/media/lq/C13E-1ED0/dataset/THUMOS/111

python -m predict.predict -d=$dir -w=$weight -v=$video_dir