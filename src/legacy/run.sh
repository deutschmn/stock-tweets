export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=6
export LOGURU_LEVEL=INFO


python -m src.experiments.regressor_concat --config=configs/concat.yaml --data.time_lag=-1 --data.min_tweets_day=5
python -m src.experiments.regressor_concat --config=configs/concat.yaml --data.time_lag=0 --data.min_tweets_day=5
python -m src.experiments.regressor_concat --config=configs/concat.yaml --data.time_lag=1 --data.min_tweets_day=5

python -m src.experiments.classifier_concat --config=configs/concat.yaml --data.time_lag=-1 --data.max_lag=null
python -m src.experiments.classifier_concat --config=configs/concat.yaml --data.time_lag=0 --data.max_lag=null
python -m src.experiments.classifier_concat --config=configs/concat.yaml --data.time_lag=1 --data.max_lag=null

python -m src.experiments.classifier_concat --config=configs/concat.yaml --data.time_lag=null --data.max_lag=5
python -m src.experiments.classifier_concat --config=configs/concat.yaml --data.time_lag=null --data.max_lag=10
python -m src.experiments.classifier_concat --config=configs/concat.yaml --data.time_lag=null --data.max_lag=20

python -m src.experiments.classifier_concat --config=configs/concat.yaml --model.transformer_config.class_path=src.model.transformer.TwitterRobertaBaseSentiment
python -m src.experiments.classifier_concat --config=configs/concat.yaml --model.transformer_config.class_path=src.model.transformer.AlbertBase

python -m src.experiments.classifier_attention --config=configs/attention.yaml --data.time_lag=0 --data.max_lag=null
python -m src.experiments.classifier_attention --config=configs/attention.yaml --data.time_lag=1 --data.max_lag=null
python -m src.experiments.classifier_attention --config=configs/attention.yaml --data.time_lag=2 --data.max_lag=null
python -m src.experiments.classifier_attention --config=configs/attention.yaml --data.time_lag=null --data.max_lag=1
python -m src.experiments.classifier_attention --config=configs/attention.yaml --data.time_lag=null --data.max_lag=2
python -m src.experiments.classifier_attention --config=configs/attention.yaml --data.time_lag=null --data.max_lag=5