accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B \
    --tasks mmlu \
    --batch_size auto:4 \
    --write_out \
    --output_path eval \


#    --tasks mmlu,mmlu_generative,pubmedqa,wikitext\


