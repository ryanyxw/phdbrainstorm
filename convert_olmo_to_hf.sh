

python FlexMoE/src/examples/huggingface/convert_checkpoint_to_hf.py \
  --checkpoint-input-path /root/ryanwang/phdbrainstorm/models/olmo2_1b-pretrain-mose-natural-1025/step30995 \
  --max-sequence-length 4096 \
  --huggingface-output-dir /root/ryanwang/phdbrainstorm/models/olmo2_1b-pretrain-mose-natural-1025/step30995-hf \
  --dtype float32