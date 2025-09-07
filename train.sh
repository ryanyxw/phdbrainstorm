litgpt pretrain meta-llama/Meta-Llama-3-8B \
   --tokenizer_dir meta-llama/Meta-Llama-3-8B \
   --data LitData \
   --data.path data/tokenized_pubmed \
   --train.lr_warmup_steps=100 \
   --out_dir models/test_pretrain \
   --resume auto \
