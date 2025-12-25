# export SUMMARY_MODEL_PATH="/path/Qwen2.5-72B-Instruct"
export MAX_LENGTH=$((1024 * 31 - 500))

cd src

# The arguments are the model path, the dataset name, and the location of the prediction file.
# bash run.sh <model_path> <dataset> <output_path>

# Dataset names (strictly match the following names):
# - gaia
# - browsecomp_zh (Full set, 289 Cases)
# - browsecomp_en (Full set, 1266 Cases)
# - xbench-deepsearch

bash sglang_model.sh /mnt/lc_share/modelscope/models/Qwen/WebAgent/WebSailor-3B/ gaia ./output/