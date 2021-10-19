export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=6

python -m src.experiments.regressor_concat --config=configs/concat.yaml --data.time_lag=-1 --data.min_tweets_day=5

python -m src.experiments.regressor_concat --config=configs/concat.yaml --data.time_lag=0 --data.min_tweets_day=5

python -m src.experiments.regressor_concat --config=configs/concat.yaml --data.time_lag=1 --data.min_tweets_day=5