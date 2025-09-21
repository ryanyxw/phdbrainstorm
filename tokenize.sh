jsonl_file="data/pubmed_orig/out.jsonl"
destination="data/pubmed_orig/tokenized"
tokenizer_name="allenai/OLMo-2-1124-7B"

n=64

# shard it into n files
split -l $(( $(wc -l < $jsonl_file) / n )) -d --additional-suffix=.jsonl $jsonl_file shard_

# gzip each shard
for f in shard_*.jsonl; do gzip "$f"; done

# tokenize the files

dolma tokens \
  --documents shard_*.jsonl.gz \
  --tokenizer.name_or_path ${tokenizer_name} \
  --destination ${destination} \
  --processes ${n} \