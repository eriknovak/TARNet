#!/bin/bash

# stop whole script when CTRL+C
trap "exit" INT

# ===============================================
# Prepare project environment
# ===============================================

if [ -d "./venv" ]; then
    . ./venv/bin/activate
else
    echo "Python environemnt does not exist"
    exit
fi;

# ===============================================
# Prepare dataset names
# ===============================================


declare -a DATASETS=(
    "Adiac"
    #"ArrowHead"
    #"Chinatown"
    #"ChlorineConcentration"
    # "CinCECGTorso" # CUDA OUT OF MEMORY
    #"DiatomSizeReduction"
    #"Earthquakes"
    #"ECG200"
    #"ECG5000"
    #"ECGFiveDays"
    #"FaceAll"
    #"FiftyWords"
    #"FordA"
    #"FordB"
    #"Haptics"
    # "InlineSkate" # CUDA OUT OF MEMORY
    #"ItalyPowerDemand"
    #Mallat"
    #"MedicalImages"
    #"MelbournePedestrian"
    #"MoteStrain"
    #"OSULeaf"
    #"Phoneme"
    #"Plane"
    #"PowerCons"
    #"SonyAIBORobotSurface1"
    #"SonyAIBORobotSurface2"
    #"Strawberry"
    #"SwedishLeaf"
    #"Symbols"
    #"TwoLeadECG"
    #"Wafer"
    #"WordSynonyms"
    #"Yoga"
)

for DATASET in "${DATASETS[@]}"; do
    echo "EVALUATION ON DATASET: $DATASET"
    if [ "$(ls ./data/raw/$DATASET/$DATASET_*.txt | wc -l)" -ge "1" ]; then
        python ./scripts/train.py --dataset $DATASET --lr 0.001 --batch 64 --epochs 100 --sparsity 0.8
    else
        echo "NO TRAIN OR TEST SET FOUND FOR DATASET: $DATASET"
    fi
done;
