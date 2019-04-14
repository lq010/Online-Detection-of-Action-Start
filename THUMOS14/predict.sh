
#dir where, by default, the weights are stored
#the predictions will be stored in the same folder
dir=/media/lq/C13E-1ED0/dataset/UCF_Crimes/results/adam_c3d_pretrain_A_BG_11_rr/weights/

#the name of the weight file (*.hdf5)
weight=weights.03-0.688.hdf5

#video path (test videos)
video_dir=/media/lq/C13E-1ED0/dataset/THUMOS/test/

python -m predict.predict -d=$dir -w=$weight -v=$video_dir 