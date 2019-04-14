ground_truth=/home/lq/Documents/Thesis/Thesis/UCF/data/ucf_accident_test_ground_truth.json

#rpath of the result(json file)

#adaptive
# final_result=/media/lq/C13E-1ED0/dataset/THUMOS/result/adam_2epoch/best_weight/predictions_adam_score_0.5.json

#adaptive + TC
# final_result=/home/lq/Documents/Thesis/Thesis/results/adam_temporal_8/weights/predictions_weights.02-2.111_score_0.5.json

#adaptive + TC + GAN
# final_result=/home/lq/Documents/Thesis/Thesis/results/gan_7/weights/predictions_c3d_TC_GAN_21_outputs_it1500_score_0.5.json

final_result=/media/lq/C13E-1ED0/dataset/UCF_Crimes/results/adam_c3d_final_1_pre_train_1M/weights/ucf_predictions_weights.01-0.597_score_0.5.json 
python -m Evaluation.get_detection_performance  $ground_truth $final_result
python -m Evaluation.get_detection_performance  $ground_truth $final_result >> $final_result.txt