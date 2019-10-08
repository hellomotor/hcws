argc=$#
if [ $argc -lt 1 ]; then
  echo "Usage: $0 <dataset>"
  exit 1
fi
DATASET=$1

ROOT_DIR=$HOME/git_repo/hcws
DATA_DIR=$ROOT_DIR/corpus/$DATASET
TRAIN_DIR=$DATA_DIR/train
BERT_DIR=$HOME/git_repo/tf_ner/bert_lstm_crf/chinese_L-12_H-768_A-12

cd $ROOT_DIR
cmd="python hcws/train.py \
	--train_data_path=$TRAIN_DIR/train.tf_record \
	--test_data_path=$TRAIN_DIR/eval.tf_record \
	--batch_size=16 \
	--label_dict_path=$TRAIN_DIR/tags.dic \
	--max_sequence_length=200 \
	--ckpt_dir=$TRAIN_DIR/logs_bert \
	--bert_config_file=$BERT_DIR/bert_config.json \
	--vocab_file=$BERT_DIR/vocab.txt \
	--init_checkpoint=$BERT_DIR/bert_model.ckpt"
echo $cmd
eval $cmd
