# directory of the raw predictions
dir='/home/lq/Documents/Thesis/Thesis/results/gan_r2/weights/'
# name of the prediction file
rawPredictionFile='predictions_c3d_TC_GAN_21_outputs_it1500.hdf5'
# threadhold
threadshold=0.5

python -m predict.create_pretiction_json_file -d=$dir -p=$rawPredictionFile -s=$threadshold