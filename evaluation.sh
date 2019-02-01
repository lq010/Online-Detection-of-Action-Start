ground_truth=./data/test_ground_truth.json

#rpath of the result(json file)
final_result=/Volumes/Untitled/result/adam_2epoch/video_features_score_0.5.json


python -m Evaluation.get_detection_performance  $ground_truth $final_result