# directory of the raw predictions
dir='/media/lq/C13E-1ED0/dataset/UCF_Crimes/results/adam_c3d_final_1_pre_train_1M/weights/'
# name of the prediction file
rawPredictionFile='ucf_predictions_weights.01-0.597.hdf5'
# threadhold
threadshold=0.5

python -m predict.ucf_create_pretiction_json_file -d=$dir -p=$rawPredictionFile -s=$threadshold