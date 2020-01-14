argc=$#
if [ $argc -lt 1 ]; then
  echo "Usage: $0 <dataset>"
  exit 1
fi
MODEL=bert
DATASET=$1
USE_CRF=False
ROOT_DIR=$HOME/git_repo/hcws
DATA_DIR=$ROOT_DIR/corpus/$DATASET
TRAIN_DIR=$DATA_DIR/train
# BERT_DIR=$HOME/git_repo/tf_ner/bert_lstm_crf/chinese_L-12_H-768_A-12
BERT_DIR=$HOME/git_repo/OpenCorpus/bert_model/chinese_L-12_H-768_A-12
CKPT_DIR=logs_$MODEL

if [ ! -f $TRAIN_DIR/$CKPT_DIR ]; then
	mkdir -p $TRAIN_DIR/$CKPT_DIR
fi

cd $ROOT_DIR
cmd="python hcws/train.py \
	--train_data_path=$TRAIN_DIR/train.tf_record \
	--test_data_path=$TRAIN_DIR/eval.tf_record \
	--batch_size=16 \
	--label_dict_path=$TRAIN_DIR/tags.dic \
    --log_step_count_steps=1000 \
	--max_sequence_length=200 \
	--ckpt_dir=$TRAIN_DIR/$CKPT_DIR \
	--bert_config_file=$BERT_DIR/bert_config.json \
	--vocab_file=$BERT_DIR/vocab.txt \
	--init_checkpoint=$BERT_DIR/bert_model.ckpt \
	--use_crf=$USE_CRF"
echo $cmd
eval $cmd
