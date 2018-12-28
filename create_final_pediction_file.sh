# directory of the raw predictions
dir='/home/lq/Documents/Thesis/Thesis/results/adam_temporal_6/best_weight/'
# name of the prediction file
rawPredictionFile='predictions_weights.02-2.20.hdf5'
# threadhold
threadshold=0.4

python -m predict.create_pretiction_json_file -d=$dir -p=$rawPredictionFile -s=$threadshold