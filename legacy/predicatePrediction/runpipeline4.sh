#!/bin/bash

# This shell script invokes the sequence of Python scripts that
# represents PPNN workflow 4.

#################################################################

# gather arguments from the command line

trc_id="$1"
if [ -z "$trc_id" ]; then
	echo "error: 1st argument required: trc_id"
    return 1
fi

prc_id="$2"
if [ -z "$prc_id" ]; then
	echo "error: 2nd argument required: prc_id"
    return 1
fi

rrc_id="$3"
if [ -z "$rrc_id" ]; then
	echo "error: 3rd argument required: rrc_id"
    return 1
fi

platform="$4"
if [ -z "$platform" ]; then
	echo "error: 4th argument required: platform"
    return 1
fi

workdir="$5"
if [ -z "$workdir" ]; then
	echo "error: 5th argument required: workdir"
    return 1
fi

#################################################################

curDateTime=$(date)
startTime=$(date +%s)

# write meta info to console

echo "PPNN pipeline v4 running ..."
echo "current date/time:" $curDateTime
echo ""
echo "============================================"
echo ""
echo "trc_id:" $trc_id
echo "prc_id:" $prc_id
echo "rrc_id:" $rrc_id
echo "platform:" $platform
echo "work dir:" $workdir
echo ""
echo "============================================"

# activate target conda environment
conda activate aires6

# capture shell script directory
scriptDir=$(pwd)

# change to directory containing our Python script(s)
cd ../Repo-VRD-KG/predicatePrediction/
pythonDir=$(pwd)



# step 2: train PPNN model
echo ""
echo "step 2: train PPNN model"
python vrd_ppnn_training_7.py $trc_id $platform $workdir
exit_status=$?
if [ "${exit_status}" -ne 0 ]; then
	echo ""
	echo "PPNN workflow shell script got failure exit code: $exit_status"
	echo "Shell script stopped prematurely"
    return 1
fi


#echo ""
#echo "forced stop after step 2"
#cd "$scriptDir"
#return 1


# step 3: consolidate training/validation scores
echo ""
echo "step 3: consolidate training/validation scores into .csv file"
cd ~/research/$workdir
grep -R 'avg training loss per mb' *_ckpt_training_log* | sort > vrd_ppnn_ckpt_training_loss_per_epoch.txt
grep -R 'avg validation loss per mb' *_ckpt_training_log* | sort > vrd_ppnn_ckpt_training_loss_per_epoch_val_loss.txt
grep -R 'validation global_recall@N' *_ckpt_training_log* | sort > vrd_ppnn_ckpt_training_loss_per_epoch_val_perf.txt
cd "$pythonDir"
python vrd_ppnn_loss_text2csv_3.py $trc_id $platform $workdir
exit_status=$?
if [ "${exit_status}" -ne 0 ]; then
	echo ""
	echo "PPNN workflow shell script got failure exit code: $exit_status"
	echo "Shell script stopped prematurely"
    return 1
fi



# step 4: consolidate logits captured for designated images
echo ""
echo "step 4: consolidate logits"
python vrd_ppnn_consolidate_logits.py $trc_id $platform $workdir
exit_status=$?
if [ "${exit_status}" -ne 0 ]; then
	echo ""
	echo "PPNN workflow shell script got failure exit code: $exit_status"
	echo "Shell script stopped prematurely"
    return 1
fi



# step 5: perform inference on test set 
echo ""
echo "step 5: perform inference on test set"
python vrd_ppnn_inference_2.py $trc_id $platform $workdir
exit_status=$?
if [ "${exit_status}" -ne 0 ]; then
	echo ""
	echo "PPNN workflow shell script got failure exit code: $exit_status"
	echo "Shell script stopped prematurely"
    return 1
fi



# step 6: select predictions
echo ""
echo "step 6: select predictions"
python vrd_ppnn_prediction_selection_2.py $trc_id $prc_id $platform $workdir
exit_status=$?
if [ "${exit_status}" -ne 0 ]; then
	echo ""
	echo "PPNN workflow shell script got Python failure exit code: $exit_status"
	echo "Shell script stopped prematurely"
    return 1
fi



# step 7: evaluate performance
echo ""
echo "step 7: evaluate performance"
python vrd_performance_evaluation_3.py $trc_id $prc_id $rrc_id 25 $platform $workdir
exit_status=$?
if [ "${exit_status}" -ne 0 ]; then
	echo ""
	echo "PPNN workflow shell script got failure exit code: $exit_status"
	echo "Shell script stopped prematurely"
    return 1
fi
python vrd_performance_evaluation_3.py $trc_id $prc_id $rrc_id  50 $platform $workdir
python vrd_performance_evaluation_3.py $trc_id $prc_id $rrc_id 100 $platform $workdir
python vrd_performance_evaluation_3.py $trc_id $prc_id $rrc_id 999 $platform $workdir


# step 8: consolidate performance scores
echo ""
echo "step 8: consolidate performance scores"
python vrd_consolidate_job_results_3.py $trc_id $prc_id $rrc_id 25 $platform $workdir
exit_status=$?
if [ "${exit_status}" -ne 0 ]; then
	echo ""
	echo "PPNN workflow shell script got failure exit code: $exit_status"
	echo "Shell script stopped prematurely"
    return 1
fi
python vrd_consolidate_job_results_3.py $trc_id $prc_id $rrc_id  50 $platform $workdir
python vrd_consolidate_job_results_3.py $trc_id $prc_id $rrc_id 100 $platform $workdir
python vrd_consolidate_job_results_3.py $trc_id $prc_id $rrc_id 999 $platform $workdir


# return to the shell script directory
cd "$scriptDir"


echo ""
echo "============================================"
echo ""

echo "PPNN pipeline v4 shell script all done!"
curDateTime=$(date)
echo "current date/time:" $curDateTime
endTime=$(date +%s)
elapsedSeconds=$(($endTime - $startTime))
echo "elapsed seconds:" $elapsedSeconds
elapsedMinutes=$(($elapsedSeconds / 60))
echo "elapsed minutes:" $elapsedMinutes
elapsedHours=$(($elapsedMinutes / 60))
remainingMinutes=$(($elapsedMinutes % 60))
echo "elapsed hours  :" $elapsedHours "hours" $remainingMinutes "minutes"
echo "workdir:" $workdir
echo ""



