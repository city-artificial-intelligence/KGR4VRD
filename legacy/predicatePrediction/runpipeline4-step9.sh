#!/bin/bash

# This shell script invokes step 9 of the PPNN pipeline 4.

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

dirname_pattern="$5"
if [ -z "$dirname_pattern" ]; then
	echo "error: 5th argument required: dirname_pattern"
    return 1
fi

#################################################################

curDateTime=$(date)
startTime=$(date +%s)

# write meta info to console

echo "PPNN pipeline v4 running ... step 9 ..."
echo "current date/time:" $curDateTime
echo ""
echo "============================================"
echo ""
echo "trc_id:" $trc_id
echo "prc_id:" $prc_id
echo "rrc_id:" $rrc_id
echo "platform:" $platform
echo "dir name pattern:" $dirname_pattern
echo ""
echo "============================================"

# activate target conda environment
conda activate aires6

# capture shell script directory
scriptDir=$(pwd)

# change to directory containing our Python script(s)
cd ../Repo-VRD-KG/predicatePrediction/
pythonDir=$(pwd)


# step 9: consolidate experiment cell results 
echo ""
echo "step 9: consolidate experiment cell results"
python vrd_consolidate_experiment_cell_results_1.py $trc_id $prc_id $rrc_id $platform $dirname_pattern
exit_status=$?
if [ "${exit_status}" -ne 0 ]; then
	echo ""
	echo "PPNN workflow shell script got failure exit code: $exit_status"
	echo "Shell script stopped prematurely"
    return 1
fi


# return to the shell script directory
cd "$scriptDir"


echo ""
echo "============================================"
echo ""

echo "PPNN pipeline v4 step 9 shell script all done!"
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
echo ""



