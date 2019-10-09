argc=$#
if [ $argc -lt 1 ]; then
  echo "Usage: $0 <dataset>"
  exit 1
fi
BATCH_SIZE=16
DATASET=$1
USE_CRF=True
ROOT_DIR=$HOME/git_repo/hcws
DATA_DIR=$ROOT_DIR/corpus/$DATASET
TRAIN_DIR=$DATA_DIR/train
BERT_DIR=$HOME/git_repo/tf_ner/bert_lstm_crf/chinese_L-12_H-768_A-12
EXPORT_DIR_BASE=

USE_TPU=False
TPU_ZONE=
TPU_NAME=
GCP_PROJECT=
MASTER=
NUM_CORES=8

cmd="python -m hcws.tpu.train \
	--train_data_path=$TRAIN_DIR/train.tf_record \
	--test_data_path=$TRAIN_DIR/eval.tf_record \
	--label_dict_path=$TRAIN_DIR/tags.dic \
	--train_batch_size=$BATCH_SIZE \
	--eval_batch_size=$BATCH_SIZE \
	--predict_batch_size=$BATCH_SIZE \
	--ckpt_dir=$TRAIN_DIR/logs_bert_tpu \
	--bert_config_file=$BERT_DIR/bert_config.json \
	--vocab_file=$BERT_DIR/vocab.txt \
	--init_checkpoint=$BERT_DIR/bert_model.ckpt \
	--use_crf=$USE_CRF \
	--do_train=True \
	--do_eval=True \
	--use_tpu=False  \
	--tpu_name=$TPU_NAME \
	--tpu_zone=$TPU_ZONE \
	--gcp_project=$GCP_PROJECT
	--master=$MASTER \
	--num_tpu_cores=$NUM_CORES \
	--export_dir_base=$EXPORT_DIR_BASE"
echo $cmd
eval $cmd
