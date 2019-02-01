# directory of the raw predictions
dir='/Volumes/Untitled/result/'
# name of the prediction file
rawPredictionFile='adam_2epoch/video_features.hdf5'
# threadhold
threadshold=0.5

python -m predict.create_pretiction_json_file -d=$dir -p=$rawPredictionFile -s=$threadshold