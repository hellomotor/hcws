saved_model_cli convert \
--dir "$HOME/git_repo/hcws/models/saved_model/hcws/" \
--output_dir "$HOME/git_repo/hcws/models/trt/" \
--tag_set serve \
tensorrt --precision_mode FP32 --max_batch_size 128 --is_dynamic_op True
