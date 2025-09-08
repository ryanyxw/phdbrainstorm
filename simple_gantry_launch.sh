gantry run \
  --workspace ai2/flex2 \
  --cluster   ai2/jupiter \
  --budget    ai2/oe-base \
  --priority  low \
  --weka=oe-training-default:/data/input/   \
  -- bash -c "echo hello"