ground_truth=./data/test_ground_truth.json

#rpath of the result(json file)
final_result=/home/lq/Documents/Thesis/Thesis/results/adam_temporal_6/best_weight/predictions_weights.02-2.20_score_0.5.json 


python -m Evaluation.get_detection_performance  $ground_truth $final_result