
#the director where sotre the weight, the prediction will be stored in the same folder
dir=/home/lq/Documents/Thesis/Thesis/results/gan_r2/weights/
#the name of the weight (*.hdf5)s
weight=c3d_TC_GAN_21_outputs_it1500.hdf5
#the path of the videos
video_dir=/media/lq/C13E-1ED0/dataset/THUMOS/test/

python -m predict.predict -d=$dir -w=$weight -v=$video_dir 