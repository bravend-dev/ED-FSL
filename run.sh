#!/usr/bin/env bash


source activate py36


function baseline() {
    CUDA=$1
    MODEL=$2
    ENCODER=$3
    N=5
    K=5
    OPTIMIZER=sgd
    LR=0.001
    DATASET=ace
    ALPHA=0
    BETA=0

    if [ $DATASET == "rams" || $DATASET == "fed" ] ; then
        TN=20
    else
        TN=18
    fi
    for MODEL in proto attproto matching relation ; do
        LOG=logs/${DATASET}/baseline/${N}-${K}.${OPTIMIZER}-${LR:2:10}.${MODEL}.${ENCODER}.txt
        CUDA_VISIBLE_DEVICES=$CUDA python fsl.py --dataset $DATASET \
                                            --optimizer $OPTIMIZER \
                                            --lr $LR \
                                            --train_way $TN \
                                            --alpha $ALPHA \
                                            --beta $BETA \
                                            --model $MODEL \
                                            --encoder $ENCODER \
                                            -n $N -k $K > $LOG #2> /dev/null
    done
}


function fsl() {
    CUDA=$1
    N=5
    K=5
    OPTIMIZER=sgd
    LR=0.001
    TREE=$2
    DATASET=fed
    ALPHA=$3
#    BETA=$4
    ENCODER=gcn

    if [ $DATASET == "rams" || $DATASET == "fed"  ] ; then
        TN=20
    else
        TN=18
    fi
    for BETA in 0.001 0.01; do
      LOG=logs/${DATASET}/${ENCODER}/${N}-${K}.${OPTIMIZER}-${LR:2:10}.${TREE}.alpha-${ALPHA}.bertlinear-${BETA:2:10}.txt
      CUDA_VISIBLE_DEVICES=$CUDA python fsl.py --dataset $DATASET \
                                              --optimizer $OPTIMIZER \
                                            --lr $LR \
                                            --train_way $TN \
                                            --alpha $ALPHA \
                                            --beta $BETA \
                                            --tree $TREE \
                                            --encoder $ENCODER \
                                            -n $N -k $K > $LOG #2> /dev/null
    done
}

function supervise(){
  CUDA=$1
  ENCODER=$2
  CUDA_VISIBLE_DEVICES=$CUDA python supervise.py --encoder $ENCODER > logs/supervise.${ENCODER}.adadelta.txt

}


#supervise 0 gcn &
#supervise 2 bertlinear &

