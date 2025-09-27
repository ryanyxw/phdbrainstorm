

python src/examples/huggingface/convert_checkpoint_to_hf.py \
  --checkpoint-input-path ../olmoe-pretrain-replicate/step30995/model_and_optim \
  --max-sequence-length 4096 \
  --huggingface-output-dir models/olmoe-pretrain-replicate/step30995-hf