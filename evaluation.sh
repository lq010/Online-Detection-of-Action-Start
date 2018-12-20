ground_truth=./data/test_ground_truth.json

#rpath of the result(json file)
final_result='/media/lq/C13E-1ED0/dataset/THUMOS/aaa/video_features_adam_bsize_24_f64_score_0.5.json'

python -m Evaluation.get_detection_performance  $ground_truth $final_result