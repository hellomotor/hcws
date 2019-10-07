python ~/git_repo/kcws/tools/freeze_graph.py --input_graph logs_bert/graph.pbtxt --input_checkpoint logs_bert/model.ckpt-189940 --output_node_names "ReverseSequence_1" --output_graph "./hcws_bert.pb"
