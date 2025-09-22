jsonl_file="data/pubmed_orig/out_formatted.jsonl"
sharded_jsonl_file="data/pubmed_orig/shard_"
destination="data/pubmed_orig/tokenized"
tokenizer_name="allenai/OLMo-2-1124-7B"

n=64

# shard it into n files with progress bar
#split -l $(( $(wc -l < $jsonl_file) / n )) -d --additional-suffix=.jsonl $jsonl_file $sharded_jsonl_file
#
## gzip each shard
#for f in ${sharded_jsonl_file}*.jsonl; do gzip $f; done
#
# tokenize the files

dolma tokens \
  --documents ${sharded_jsonl_file}*.jsonl.gz \
  --tokenizer.name_or_path ${tokenizer_name} \
  --tokenizer.eos_token_id 100257 \
  --tokenizer.pad_token_id 100277 \
  --destination ${destination} \
  --dtype uint32 \
  --processes ${n}