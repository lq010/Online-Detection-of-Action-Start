# python -m predict.predict -w=/media/lq/C13E-1ED0/dataset/THUMOS/result/adam_temporal/best_result/adam_temporal/weights.hdf5 -o=/media/lq/C13E-1ED0/dataset/THUMOS/result/adam_temporal -d=/media/lq/C13E-1ED0/dataset/THUMOS/test
weight=/media/lq/C13E-1ED0/dataset/THUMOS/result/adam_2epoch/best_weight/weights.hdf5
out_put_dir=/media/lq/C13E-1ED0/dataset/THUMOS/aaa
file_name=video_features_adam_bsize_24_f64.hdf5
video_dir=/media/lq/C13E-1ED0/dataset/THUMOS/aaa

python -m predict.predict -w=$weight -o=$out_put_dir -f=$file_name -d=$video_dir