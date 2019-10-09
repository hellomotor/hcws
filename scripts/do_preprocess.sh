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
cmd="python hcws/data_utils.py \
  --data_dir=$DATA_DIR/data \
  --vocab_file=$BERT_DIR/vocab.txt \
  --output_dir=$TRAIN_DIR \
  --batch_size=16 \
  --num_train_epochs=10 \
  --do_lower_case=True"
echo $cmd
eval $cmd
