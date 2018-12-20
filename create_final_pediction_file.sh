rawPredictions='/media/lq/C13E-1ED0/dataset/THUMOS/aaa/video_features_adam_bsize_24_f64.hdf5'
savePath='/media/lq/C13E-1ED0/dataset/THUMOS/aaa/'
threadshold=0.5

python -m predict.create_pretiction_json_file -p=$rawPredictions -o=$savePath -s=$threadshold