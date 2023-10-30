#!/bin/bash

do_export()
{
export $1=$2
echo export $1=$2
}

#Input Parameters

# TASK_NAME
if [ -z "$1" ]; then
  echo Usage: run_hd.sh TASK_NAME TEST_NAME GPT_OPTION [OUTPUT_FOLDER]
  echo Please set 1st parameter as task name, which is one of below:
  echo   qags_cnndm
  echo   qags_xsum
  echo   summeval
  exit 1
fi
do_export TaskName $1

# TEST_NAME
if [ -z "$2" ]; then
  echo Please set a test run name where the test result will be stored
  exit 2
fi
# descriptive name for the test
# set TestName and TestRunName
do_export TestRunName $2
do_export TestName ${TestRunName}

# GPT_OPTION
if [ -z "$3" ]; then
  echo 'Please set a config setting for how to call openai gpt: (eg. one of the key values from aoai_config.json, like 'gpt-4-32k')'
  exit 2
fi
do_export AoaiConfigSetting $3

# OUTPUT_FOLDER
if [ -z "$4" ]; then
  do_export OutputFolder CoNLI/output/$TaskName/$TestName/
else
  do_export OutputFolder $4/
fi

if [ "${TaskName}" = "qags_cnndm" ]; then
  do_export hyp ./CoNLI/test_suite/qags_cnndm/qags_cnndm_raw_response.tsv
  do_export src ./CoNLI/test_suite/qags_cnndm/src/
  do_export has_ground_truth true #  # whether ground truth summary is provided for evaluation
  do_export gpt_batch_size 1

elif [ "${TaskName}" = "qags_xsum" ]; then
  do_export hyp ./CoNLI/test_suite/qags_xsum/qags_xsum_raw_response.tsv
  do_export src ./CoNLI/test_suite/qags_xsum/src/
  do_export has_ground_truth true
  do_export gpt_batch_size 1

elif [ "${TaskName}" = "summeval" ]; then
  do_export hyp ./CoNLI/test_suite/summeval/summeval_raw_response.tsv
  do_export src ./CoNLI/test_suite/summeval/src/
  do_export has_ground_truth true
  do_export gpt_batch_size 1

else
  echo Unknown task name $TaskName
  exit 3

fi

# HD Module Behavior
do_export entity_detector text_analytics
do_export sentence_selector rule_based
do_export ta_config_setting ta-general

# AOAI Configuration Information
do_export aoai_config_setting ${AoaiConfigSetting}

# Experiment E2E Run Speed
do_export max_parallel_data 2
do_export max_parallel 2

# On-screen display behavior
do_export log_level error
do_export use_simple_progressbar True

# Debug parameters
# set to 1 to run a simple smoking test
do_export test_mode 0


# Run the HD module
# expecting 3 parameters: input_hypothesis, input_transcripts, output_folder
do_hallucination_detection()
{
    echo Running Hallucination Detection ...
    echo Task: ${TaskName}

    my_hyp=$1
    my_src=$2
    my_output_folder=$3

    echo "Start Time: $(date)" > ./CoNLI/baseline/$TaskName/details.txt
    echo "" >> ./CoNLI/baseline/$TaskName/details.txt

    python -m CoNLI.run_hallucination_detection \
         --aoai_config_setting $aoai_config_setting \
         --entity_detector_type $entity_detector \
         --ta_config_setting $ta_config_setting \
         --sentence_selector_type $sentence_selector \
         --max_parallel_data $max_parallel_data \
         --max_parallelism $max_parallel \
         --simple_progress_bar $use_simple_progressbar \
         --log_level $log_level \
         --test_mode $test_mode \
         --input_hypothesis $my_hyp \
         --input_src $my_src \
         --output_folder $my_output_folder \
         --gpt_batch_size $gpt_batch_size

    echo "Label: $TestName" >> ./CoNLI/baseline/$TaskName/details.txt
    echo "Hypothesis: $my_hyp"  >> ./CoNLI/baseline/$TaskName/details.txt
    echo "sources: $my_src" >> ./CoNLI/baseline/$TaskName/details.txt
    echo "Output folder: $my_output_folder" >> ./CoNLI/baseline/$TaskName/details.txt
    echo "" >> ./CoNLI/baseline/$TaskName/details.txt
    echo "AOAI Config Setting: $aoai_config_setting" >> ./CoNLI/baseline/$TaskName/details.txt
    echo "Entity Detector Type: $entity_detector" >> ./CoNLI/baseline/$TaskName/details.txt
    echo "TA Config Setting: $ta_config_setting" >> ./CoNLI/baseline/$TaskName/details.txt
    echo "Sentence Selector Type: $sentence_selector" >> ./CoNLI/baseline/$TaskName/details.txt
    echo "Max Parallel Encounters: $max_parallel_data" >> ./CoNLI/baseline/$TaskName/details.txt
    echo "Max Parallelism: $max_parallel" >> ./CoNLI/baseline/$TaskName/details.txt
    echo "Test Mode: test_mode $test_mode" >> ./CoNLI/baseline/$TaskName/details.txt
    echo "GPT Batch Size: $gpt_batch_size" >> ./CoNLI/baseline/$TaskName/details.txt
    echo "" >> ./CoNLI/baseline/$TaskName/details.txt
}


# calculate the metrics
calculate_metrics()
{
    echo Calculating Metrics ...
    my_hyp=$1
    my_output_folder=$2

    if [ "$has_ground_truth" = true ]; then
      python -m CoNLI.run_hallucination_sentence_gt_evaluator --gtfile $my_hyp --hd_result_folder $my_output_folder --output_folder $my_output_folder
    fi

    # TODO: fix this
    #hallucinationjsonl="${my_output_folder}/hallucinations/allhallucinations.jsonl"
    #python -m CoNLI.run_response_quality_evaluator --hallucinationjsonl $hallucinationjsonl
}

# Copy metrics and run details to task folder
record_testrun_details()
{
  echo Recording test run details ...

  mkdir -p ./CoNLI/baseline/$TaskName/
  mkdir -p ./CoNLI/baseline/$TaskName/Breakdown

  cp $(ls ${OutputFolder}/intermediate/Analysis.Results.*.txt | grep -v Analysis.Results.OVERALL.txt) ./CoNLI/baseline/$TaskName/Breakdown
  cp ${OutputFolder}/intermediate/Analysis.Results.OVERALL.txt ./CoNLI/baseline/$TaskName/

  echo End Time: $(date) >> ./CoNLI/baseline/$TaskName/details.txt
}

do_hallucination_detection $hyp $src $OutputFolder
calculate_metrics $hyp ${OutputFolder}
record_testrun_details
