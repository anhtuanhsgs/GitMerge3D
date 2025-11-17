export PYTHONPATH=./

config=$1
CONFIG_NAME=${config}
EXP_ROOT_FOLDER=semseg-sonata-v1m1-0-base-0c-scannet-ft

PTV3_CONFIG_PATH=./configs/sonata/${CONFIG_NAME}.py
PTV3_WEIGHTS_PATH=./exp/sonata/${EXP_ROOT_FOLDER}/model/model_best.pth

PTV3_SAVE_PATH=./exp/sonata/${CONFIG_NAME}
echo "export PYTHONPATH=./ python tools/test_merge_rates.py --config-file ${PTV3_CONFIG_PATH} --options save_path=${PTV3_SAVE_PATH} weight=${PTV3_WEIGHTS_PATH}"

python tools/test_merge_rates.py --config-file ${PTV3_CONFIG_PATH} --options save_path=${PTV3_SAVE_PATH} weight=${PTV3_WEIGHTS_PATH}
