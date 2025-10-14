

python OLMo-core/src/examples/huggingface/convert_checkpoint_to_hf.py \
  --checkpoint-input-path /root/ryanwang/models/olmoe-pretrain-mose-unbalanced-1012/step30995 \
  --max-sequence-length 2048 \
  --huggingface-output-dir /root/ryanwang/phdbrainstorm/models/olmoe-pretrain-mose-unbalanced-1012/step30995-hf \
  --dtype float32