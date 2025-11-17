export PYTHONPATH=./


CONFIG_NAME=$1
EXP_ROOT_FOLDER=semseg-sonata-v1m1-0-base-3a-s3dis-lin

PTV3_CONFIG_PATH=./configs/sonata/${CONFIG_NAME}.py
PTV3_WEIGHTS_PATH=./exp/sonata/${EXP_ROOT_FOLDER}/model/model_best.pth

PTV3_SAVE_PATH=./exp/s3dis/${CONFIG_NAME}

echo "export PYTHONPATH=./ python tools/test.py --config-file ${PTV3_CONFIG_PATH} --options save_path=${PTV3_SAVE_PATH} weight=${PTV3_WEIGHTS_PATH}"

python tools/test.py --config-file ${PTV3_CONFIG_PATH} --options save_path=${PTV3_SAVE_PATH} weight=${PTV3_WEIGHTS_PATH}
