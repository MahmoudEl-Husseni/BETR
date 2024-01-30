#!/bin/bash

# : <<'case_statement'
shopt -s nocasematch
case "$1" in
        argo-1)
            exp_name="Argo-1"
            ;;
        argo-norm|argo-normalized)
            exp_name="Argo-Normalized"
            ;;
        argo-pad)
            exp_name="Argo-pad"
            ;;
        argo-avg)
            exp_name="Argo-avg"
            ;;
        argo-gnn-gnn)
            exp_name="Argo-GNN-GNN"
            ;;
        *)
            echo "not supported Experiment Name: $1"
            exit 1
            ;;
esac
shopt -u nocasematch
# case_statement



: <<'select_statement'
options=("Argo-1" "Argo-Normalized" "Argo-pad" "Argo-avg" "Argo-GNN-GNN")


PS3="Select an Experiment Name: "
select choice in "${options[@]}"; do
    case $choice in
        "Argo-1")
            exp_name="Argo-1"
            break
            ;;
        "Argo-Normalized")
            exp_name="Argo-Normalized"
            break
            ;;
        "Argo-pad")
            exp_name="Argo-pad"
            break
            ;;
        "Argo-avg")
            exp_name="Argo-avg"
            break
            ;;
        "Argo-GNN-GNN")
            exp_name="Argo-GNN-GNN"
            break
            ;;
        *) 
            echo "not supported Experiment Name: $1"
            exit 1
            ;;
    esac
done
select_statement

echo Selected Experiment: $exp_name
if [ $2 == 't' ]
then 
  shift 2
  while [[ $# -gt 0 ]]; do
    case "$1" in
        -s|--scene_name)
            export scene_name=$2
            shift 2
            ;;
        -d|--data_path)
            export data_path=$2
            shift 2
            ;;
        -best|--best_models_path)
            export best_models_path=$2
            shift 2
            ;;
        -sv|--save_path)
            export save_path=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

else 
read -p "Scene name: " scene_name
read -p "Data Path: " data_path 
read -p "Best models path: " best_models_path
read -p "Save path: " save_path
fi 

if [ -z $data_path ]
then 
data_path=$(readlink -f ../../../Argoverse\ Dataset/train_interm)
fi 

if [ -z $best_models_path ]
then 
best_models_path=$(readlink -f ../../../Argoverse\ Dataset/out/best_models)
fi 

if [ -z $$save_path ]
then 
save_path=`readlink -f ../../../Argoverse\ Dataset/out/`
fi



# echo ${exp_name}.sh -s $scene_name -d "$data_path" -best "$best_models_path" -sv "$save_path"
cd VectorNet/Argoverse2/inference

bash ${exp_name}.sh -s $scene_name -d "$data_path" -best "$best_models_path" -sv "$save_path"

echo "Inferenece status code: " $?