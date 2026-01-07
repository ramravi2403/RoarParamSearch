from models.ModelFactory import ModelFactory


class ModelWrapper:
    def __init__(self, input_dim, model_type: str):
        self.__model = ModelFactory().create_instance(model_type, input_dim)

    def train(self, X_train, y_train, num_epochs=100, lr=0.01, verbose=False):
        return self.__model.train_model(X_train, y_train, num_epochs, lr, verbose)

    def predict(self, X):
        return self.__model.predict(X)

    def extract_weights(self):
        if not hasattr(self.__model, 'extract_weights'):
            raise AttributeError(f"{self.__model.__class__.__name__} does not support weight extraction")
        return self.__model.extract_weights()

    def get_model(self):
        return self.__model