# directory of the raw predictions
dir='/media/lq/C13E-1ED0/dataset/UCF_Crimes/results/gan_190/weights/'
# name of the prediction file
rawPredictionFile='ucf_predictions_c3d_TC_GAN_1_outputs_it190.hdf5'
# threadhold
threadshold=0.5

python -m predict.create_pretiction_json_file -d=$dir -p=$rawPredictionFile -s=$threadshold