root_dir=$HOME/git_repo/hcws
ckpt_dir=$root_dir/corpus/RMRB/train/logs_bert
export_dir=$root_dir/models
ckpt=$(ls -th $ckpt_dir/model.ckpt*|head -n1|awk -F'.meta' '{print $1}')
#echo $ckpt
rm -f "$export_dir/hcws_bert.pb"
output_node_names='pred_ids,seq_lengths'
cmd="python ~/git_repo/kcws/tools/freeze_graph.py --input_graph $ckpt_dir/graph.pbtxt --input_checkpoint $ckpt --output_node_names \"$output_node_names\" --output_graph $export_dir/hcws_bert.pb"
echo $cmd
eval $cmd
