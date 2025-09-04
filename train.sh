litgpt pretrain pythia-14m \
   --tokenizer_dir meta-llama/Meta-Llama-3-8B \
   --data TextFiles \
   --data.train_data_path data/test_pretrain \
   --train.lr_warmup_steps=1 \
   --optimizer AdamW \
   --optimizer.lr 0.005 \
   --out_dir models/test_pretrain