ROOT_DIR=/root/ryanwang/phdbrainstorm
MODEL_DIR=${ROOT_DIR}/models

set -e

RUN_DIRS=(
    "olmo2_pubmed_100k_BS-128"
    "pubmed_100k_BS-128"
    "olmo2_pubmed-hashprefix-20_100k_BS-128"
    "olmo2_pubmed-hashprefix-5_100k_BS-128"
    "pubmed-hashprefix-20_100k_BS-128"
    "pubmed-hashprefix-5_100k_BS-128"
)

CKPT_DIR=(
    "."
    "checkpoint-75"
    "checkpoint-150"
    "checkpoint-225"
    "checkpoint-300"
)

# loop through the directories
for RUN_NAME in "${RUN_DIRS[@]}"; do
    RUN_DIR=${MODEL_DIR}/${RUN_NAME}
    echo "Processing directory: $RUN_DIR"

#    for ckpt in "$RUN_DIR"/checkpoint-*; do
    for ckpt_name in "${CKPT_DIR[@]}"; do
        ckpt=${RUN_DIR}/${ckpt_name}

        # convert checkpoint if not already converted
        if compgen -G "$ckpt/pytorch_model*" > /dev/null; then
            echo "Found converted checkpoint directory: $ckpt"
        else
            echo "Converting shards in $ckpt ..."
            python ${RUN_DIR}/zero_to_fp32.py $ckpt $ckpt
        fi

        # begin evaluation
        accelerate launch -m lm_eval --model hf \
          --model_args pretrained="${ckpt}" \
          --tasks wikitext \
          --batch_size auto:4 \
          --write_out \
          --output_path "${ckpt}/eval"
    done
done

#accelerate launch -m lm_eval --model hf \
#    --model_args pretrained=allenai/OLMo-2-1124-7B \
#    --tasks pubmedqa \
#    --batch_size auto:4 \
#    --write_out \
#    --output_path models/olmo2-1124-7B \


#    --model_args pretrained=meta-llama/Meta-Llama-3-8B \
#    --tasks mmlu,mmlu_generative,pubmedqa,wikitext\


