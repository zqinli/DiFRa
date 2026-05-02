CUDA_DEVICES="0"

VERSION_TAG="1"


DS_CONFIG=""
BERT_MODEL_NAME="../models/bert-base-uncased"

MAX_PROMPT_LENGTH=8192
MAX_QA_PAIR=15
INFER_MAX_QA_PAIR=10
COND_MAX_LENGTH=256
NUM_CONCEPTS=8

LOAD_MODEL_ACCURACY="bf16"
LAMBDA_DIFF=0.1
MLP_BLOCK=2
NUM_TRAIN_EPOCHS=3
PER_DEVICE_TRAIN_BATCH_SIZE=1


run_train() {
    local dataset_name=$1
    local train_data_path=$2
    local test_data_path=$3
    local model_name=$4
    local output_dir=$5
    local model_type=$6


    CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} i-cli train \
        --data-args.dataset-name "${dataset_name}" \
        --data-args.train-data-path "${train_data_path}" \
        --data-args.test-data-path "${test_data_path}" \
        --data-args.max-prompt-length "${MAX_PROMPT_LENGTH}" \
        --data-args.max-qa-pair "${MAX_QA_PAIR}" \
        --data-args.infer-max-qa-pair "${INFER_MAX_QA_PAIR}" \
        --data-args.cond-max-length "${COND_MAX_LENGTH}" \
        --data-args.num-concepts "${NUM_CONCEPTS}" \
        --data-args.use-concepts \
        --data-args.use-knowledge-graph \
        --model-args.model-type "${model_type}" \
        --model-args.model-name "${model_name}" \
        --model-args.load-model-accuracy "${LOAD_MODEL_ACCURACY}" \
        --model-args.bert-model-name "${BERT_MODEL_NAME}" \
        --model-args.lambda-diff "${LAMBDA_DIFF}" \
        --model-args.diffusion-mlp-block-num "${MLP_BLOCK}" \
        --model-args.use-diffusion \
        --model-args.use-ema \
        --finetuning-args.freeze-llm \
        --finetuning-args.use-lora \
        --finetuning-args.use-lora-on-denoiser \
        --num-train-epochs "${NUM_TRAIN_EPOCHS}" \
        --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
        --output-dir "${output_dir}" \
        --do-infer-after-train \
        --deepspeed "${DS_CONFIG}"
}


MODEL_QWEN="../models/Qwen2.5-7B-Instruct"

run_train "drop" \
    "train.json" \
    "test.json" \
    "${MODEL_QWEN}" \
    "output/${VERSION_TAG}/with_qwen/DROP" \
    "instruct"

