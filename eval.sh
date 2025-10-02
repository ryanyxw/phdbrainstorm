#bash FlexOlmo/src/scripts/eval/run_eval.sh models/olmoe-pretrain-replicate/step30995-hf mc9 models/olmoe-pretrain-replicate/step30995-hf/evals 1

MODEL=/root/ryanwang/phdbrainstorm/models/olmoe-pretrain-replicate/step30995-hf
OUTPUT_DIR=${MODEL}/evals
NUM_GPUS=$(nvidia-smi -L | wc -l)

TASKS=(
#  # mc9 mc
#  arc_easy:mc::olmes
#  arc_challenge:mc::olmes
#  boolq:mc::olmes
#  csqa:mc::olmes
#  hellaswag:mc::olmes
#  openbookqa:mc::olmes
#  piqa:mc::olmes
#  socialiqa:mc::olmes
#  winogrande:mc::olmes

#  # gen5
#  coqa::olmes
#  squad::olmes
#  naturalqs::olmes
#  triviaqa::olmes
#  drop::olmes

#	# mmlu mc
#  mmlu:mc::olmes

#  # mmlu_pro mc
#	mmlu_pro_mc::none

#	# agi_eval
#	agi_eval_english:1shot::olmes

#	# bbh hang problem
	#bbh:cot-v1::olmes

	# math2 hang problem (gsm8k)
  #gsm8k::olmes
  minerva_math_algebra::olmes
  minerva_math_counting_and_probability::olmes
  minerva_math_geometry::olmes
  minerva_math_intermediate_algebra::olmes
  minerva_math_number_theory::olmes
  minerva_math_prealgebra::olmes
  minerva_math_precalculus::olmes

  # code4
  codex_humaneval:temp0.8
  codex_humanevalplus:temp0.8
  mbpp::none
  mbppplus::none

  # mc9 rc
  arc_easy:rc::olmes
  arc_challenge:rc::olmes
  boolq:rc::olmes
  csqa:rc::olmes
  hellaswag:rc::olmes
  openbookqa:rc::olmes
  piqa:rc::olmes
  socialiqa:rc::olmes
  winogrande:rc::olmes

  # mmlu rc
#  mmlu:rc::olmes
)

for TASK in "${TASKS[@]}"; do
	# For setting the output_dir
	model=$(echo $MODEL | cut -d'/' -f2)
	# OOM with some tasks, so batch size to be 1
	if [[ $TASK == "minerva_math_"* || $TASK == "mbpp"* || $TASK == "bigcodebench"* || $TASK == "sciriff"* ]] ; then
		batch_size=1
	else
		batch_size=4
	fi

	PYTHONPATH=./FlexOlmo python FlexOlmo/src/scripts/eval/launch_eval.py \
	--model $MODEL \
	--model-type hf \
	--task $TASK \
	--limit 1000 \
	--output-dir $OUTPUT_DIR \
	--batch-size $batch_size \
	--gpus $NUM_GPUS
done