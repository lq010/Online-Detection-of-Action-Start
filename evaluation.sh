ground_truth=./data/test_ground_truth.json

#rpath of the result(json file)
final_result=/home/lq/Documents/Thesis/Thesis/results/adam/best_weight/predictions_weights_score_0.0.json

python -m Evaluation.get_detection_performance  $ground_truth $final_result