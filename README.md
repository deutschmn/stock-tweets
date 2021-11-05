# Sentiment for Price Prediction (`SePP`): Predicting Stock Performance Based on Tweets

This project predicts whether stock prices are going up or down using tweets. I implemented it in the course of my bachelor's thesis in business economics at [University of Graz](http://uni-graz.at/en/).

## Abstract

> Stock prices are inﬂuenced by investors’ beliefs, which many express on social media. As recent advances in machine learning have achieved impressive results in detecting sentiment in human language, this thesis aims to apply them to the task of stock price prediction. We present a machine learning model that takes as input tweets about certain assets, derives their sentiments and predicts whether the stock price will go up or down after that. Our model achieves a new state of the art in prediction accuracy on the StockNet data set and demonstrates that tweet sentiments have a discernible effect on stock prices.

## Usage

The project uses [PyTorch Lightning CLI](https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_cli.html). Start training for the concatenation classifier like this:

```bash
python -m src.experiments.classifier_concat --config=configs/concat.yaml
```

## Authors
[Patrick Deutschmann](mailto:patrick@deutschmann.xyz)