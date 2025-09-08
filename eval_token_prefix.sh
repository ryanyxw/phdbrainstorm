MODEL_DIR=models

set -e

RUN_DIRS=(
    "pubmed_100k_BS-32"
)

CKPT_DIR=(
    "."
)

# loop through the directories
for RUN_DIR in "${MODEL_DIR}/${RUN_DIRS[@]}"; do
    echo "Processing directory: $RUN_DIR"

#    for ckpt in "$RUN_DIR"/checkpoint-*; do
    for ckpt in "$RUN_DIR"/"${CKPT_DIR[@]}"; do

        # convert checkpoint if not already converted
        if compgen -G "$ckpt/pytorch_model*.bin" > /dev/null; then
            echo "Found converted checkpoint directory: $ckpt"
        else
            echo "Converting shards in $ckpt ..."
            python ${RUN_DIR}/zero_to_fp32.py $ckpt $ckpt
        fi

        # begin evaluation
        accelerate launch -m lm_eval --model hf \
          --model_args pretrained="${ckpt}" \
          --tasks pubmedqa \
          --batch_size auto:4 \
          --write_out \
          --output_path "${ckpt}/eval"
    done
done

#accelerate launch -m lm_eval --model hf \
#    --model_args models/pubmed_100k_BS-128/checkpoint-25 \
#    --tasks pubmedqa \
#    --batch_size auto:4 \
#    --write_out \
#    --output_path eval \


#    --model_args pretrained=meta-llama/Meta-Llama-3-8B \
#    --tasks mmlu,mmlu_generative,pubmedqa,wikitext\


