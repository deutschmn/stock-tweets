from src.model.concat import ConcatMovementPredictor
from src.model.attention import AttentionMovementPredictor
from src.model.classification import ClassificationMovementPredictor
from src.model.regression import RegressionMovementPredictor


class AttentionClassifier(AttentionMovementPredictor, ClassificationMovementPredictor):
    pass

class AttentionRegressor(AttentionMovementPredictor, RegressionMovementPredictor):
    pass

class ConcatClassifier(ConcatMovementPredictor, ClassificationMovementPredictor):
    pass