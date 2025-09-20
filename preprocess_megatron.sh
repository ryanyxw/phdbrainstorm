python NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input=data/pubmed_orig/out.jsonl \
    --json-keys=text \
    --tokenizer-library=huggingface \
    --tokenizer-type=meta-llama/Meta-Llama-3-8B \
    --dataset-impl=mmap \
    --output-prefix=data/pubmed_orig/pubmed_orig_megatron \
    --split-sentences \
    --workers=128