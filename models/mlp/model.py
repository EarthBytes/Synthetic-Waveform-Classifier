from sklearn.neural_network import MLPClassifier
from config import RANDOM_STATE, MLP_HIDDEN_LAYER_SIZES, MLP_MAX_ITER
from models.pipelines import scaled_pipeline
from sklearn.pipeline import Pipeline

def create_mlp() -> Pipeline:
    return scaled_pipeline(
        MLPClassifier(
            hidden_layer_sizes=MLP_HIDDEN_LAYER_SIZES,
            activation='relu',
            solver='adam',
            early_stopping=True,
            validation_fraction=0.1,
            max_iter=MLP_MAX_ITER,
            random_state=RANDOM_STATE
        )
    )    