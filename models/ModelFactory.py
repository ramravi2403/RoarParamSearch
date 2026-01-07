from models.SimpleClassifier import SimpleClassifier
from models.DeepClassifier import DeepClassifier


class ModelFactory:

    def create_instance(self,model_type,input_dim:int):
        if model_type == "simple":
            return SimpleClassifier(input_dim)
        elif model_type == "deep":
            return DeepClassifier(input_dim)
        else:
            raise ValueError(f"Invalid model type {model_type}")