#!/bin/bash

cd ..
branch_name=$(git branch --show-current)

# set -x

if [ $branch_name != 'argo-pad' ]
then 
	echo Current Branch:  $branch_name
	git switch argo-pad
fi 
git branch
# Parse input args
export EXPERIMENT_NAME=Argo-pad
best_models_path=''
save_path=''
data_path=''

while [[ $# -gt 0 ]]; do
    sleep 3s
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

if [ -z $data_path ]
then 
export data_path=$(readlink -f ../../Argoverse\ Dataset/train_interm)
fi 

if [ -z $best_models_path ]
then 
export best_models_path=$(readlink -f ../../Argoverse\ Dataset/out/best_models)
fi 

while [ ! -d "$best_models_path" ] 
do 
echo $best_models_path not a directory 
read -p "You have to add a valid 'best models' directory path: " best_models_path
done 

while [ ! -d "$data_path" ] 
do 
echo $data_path not a directory 
read -p "You have to add a valid 'data path' directory: " data_path
done 

export data_path=$data_path
export best_models_path=$best_models_path


if [ -z $save_path ]
then 
save_path=`readlink -f ../../Argoverse\ Dataset/out/`
export save_path=${save_path}/preds
	if [ ! -d save_path ]
	then
	  mkdir -p "$save_path"
	  echo Direcotry "$save_path" Created
	fi  
fi 
export save_path=$save_path

echo $save_path
# set +x

python3 inference/infere.py
