#!/bin/bash

source activate py37

function run(){

  jobfile=$2
  cuda=$1

  sleep $cuda

  export CUDA_VISIBLE_DEVICES=$cuda

  while true
  do
    command=$(head -n 1 $jobfile)
    if [ -z "$command" ] ; then
      break
    fi
    echo $command
    sed -i '1d' $jobfile
    eval $command
  done
}

jobfile=commands.txt

cp $jobfile  commands.arxiv

run 0 $jobfile &
# if [ $HOSTNAME == "hal.cs.uoregon.edu" ] ; then
#   run 0 $jobfile &
#   run 1 $jobfile &
# elif [ $HOSTNAME == "iq.cs.uoregon.edu" ] ; then
#   run 0 $jobfile &
#   run 1 $jobfile &
#   run 2 $jobfile &
# else
#   # Legendary server
#   run 0 $jobfile &
#   run 1 $jobfile &
#   run 2 $jobfile &
#   run 3 $jobfile &
# fi
