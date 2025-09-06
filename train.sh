litgpt pretrain pythia-14m \
   --tokenizer_dir meta-llama/Meta-Llama-3-8B \
   --data TextFiles \
   --data.train_data_path data/test_pretrain \
   --data.max_seq_len=3 \
   --train.lr_warmup_steps=1 \
   --out_dir models/test_pretrain \
   --resume auto \
