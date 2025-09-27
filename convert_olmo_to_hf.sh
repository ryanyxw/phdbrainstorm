

python OLMo-core/src/examples/huggingface/convert_checkpoint_to_hf.py \
  --checkpoint-input-path /root/ryanwang/olmoe-pretrain-replicate/step30995 \
  --max-sequence-length 4096 \
  --huggingface-output-dir /root/ryanwang/phdbrainstorm/models/olmoe-pretrain-replicate/step30995-hf \
  --moe-capacity-factor 1000