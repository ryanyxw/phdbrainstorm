TORCHDYNAMO_VERBOSE=1 litgpt pretrain meta-llama/Meta-Llama-3-8B \
   --tokenizer_dir meta-llama/Meta-Llama-3-8B \
   --data LitData \
   --data.data_path "data/tokenized_pubmed" \
   --train.lr_warmup_steps=100 \
   --train.micro_batch_size=1 \
   --out_dir models/test_pretrain \
   --resume auto \
   --precision bf16-true \
