# DATA
DATA_PATH="../machine-tasks/LookupTables/lookup-3bit/samples/sample1"
TRAIN_PATH="${DATA_PATH}/train.tsv"
VALIDATION_PATH="${DATA_PATH}/validation.tsv"
TEST1_PATH="${DATA_PATH}/heldout_compositions.tsv"
TEST2_PATH="${DATA_PATH}/heldout_inputs.tsv"
TEST3_PATH="${DATA_PATH}/heldout_tables.tsv"
TEST4_PATH="${DATA_PATH}/new_compositions.tsv"
TEST5_PATH="${DATA_PATH}/longer_compositions_incremental.tsv"
TEST6_PATH="${DATA_PATH}/longer_compositions_new.tsv"
TEST7_PATH="${DATA_PATH}/longer_compositions_seen.tsv"
TEST8_PATH="${DATA_PATH}/heldout_tables_sn.tsv"
TEST9_PATH="${DATA_PATH}/heldout_tables_ns.tsv"
MONITOR_DATA="${VALIDATION_PATH} ${TEST1_PATH} ${TEST2_PATH} ${TEST3_PATH} ${TEST4_PATH} ${TEST5_PATH} ${TEST6_PATH} ${TEST7_PATH} ${TEST8_PATH} ${TEST9_PATH}"

# TRAIN SETTINGS
TF=0.5
EPOCHS=1
BATCH_SIZE=1
EVAL_BATCH_SIZE=2000
METRICS="seq_acc"
SAVE_EVERY=100
PRINT_EVERY=100
CUDA=0
EXPT_DIR="seq2attn_lookup_checkpoints"

# MODEL PARAMETSR
DROPOUT=0
ATTENTION="pre-rnn"
ATTN_METHOD="mlp"
EMB_SIZE=512
HIDDEN_SIZE=512
RNN_CELL=lstm
ATTN_VALS=embeddings
SAMPLE_TRAIN=softmax_st
SAMPLE_INFER=softmax_st
INIT_TEMP=5
LEARN_TEMP=conditioned
FULL_ATTENTION_FOCUS=no

python train_model.py \
    --train $TRAIN_PATH \
    --dev $VALIDATION_PATH \
    --monitor $MONITOR_DATA \
    --metrics $METRICS \
    --output_dir $EXPT_DIR \
    --epochs $EPOCHS \
    --rnn_cell $RNN_CELL \
    --embedding_size $EMB_SIZE \
    --hidden_size $HIDDEN_SIZE \
    --dropout_p_encoder $DROPOUT \
    --dropout_p_decoder $DROPOUT \
    --teacher_forcing_ratio $TF \
    --attention $ATTENTION \
    --attention_method $ATTN_METHOD \
    --batch_size $BATCH_SIZE \
    --eval_batch_size $EVAL_BATCH_SIZE \
    --save_every $SAVE_EVERY \
    --print_every $PRINT_EVERY \
    --write-logs "${EXPT_DIR}_LOG" \
    --cuda_device $CUDA \
    --sample_train $SAMPLE_TRAIN \
    --sample_infer $SAMPLE_INFER \
    --initial_temperature $INIT_TEMP \
    --learn_temperature $LEARN_TEMP \
    --attn_vals $ATTN_VALS \
    --full_attention_focus $FULL_ATTENTION_FOCUS

python evaluate.py \
    --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -2 | tail -1) \
    --test_data $VALIDATION_PATH \
    --batch_size $EVAL_BATCH_SIZE \
    --attention pre-rnn \
    --attention_method mlp \
