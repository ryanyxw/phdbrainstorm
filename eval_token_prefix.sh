accelerate launch -m lm_eval --model hf \
    --model_args pretrained=meta-llama/Meta-Llama-3-8B \
    --tasks mmlu,mmlu_generative,pubmed_qa,wikitext\
    --batch_size auto:4
