jsonl_file="data/pubmed_orig/out.jsonl"
sharded_jsonl_file="data/pubmed_orig/shard_"
destination="data/pubmed_orig/tokenized"
tokenizer_name="allenai/OLMo-2-1124-7B"

n=64

# shard it into n files
split -l $(( $(wc -l < $jsonl_file) / n )) -d --additional-suffix=.jsonl $jsonl_file $sharded_jsonl_file

# gzip each shard
for f in ${sharded_jsonl_file}*.jsonl; do gzip $f; done

# tokenize the files

dolma tokens \
  --documents shard_*.jsonl.gz \
  --tokenizer.name_or_path ${tokenizer_name} \
  --destination ${destination} \
  --processes ${n}