import onnxruntime
from abc import ABC, abstractmethod

class BaseONNXModel(ABC):
    def __init__(self, path):
        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.run(image)

    def initialize_model(self, path):
        self.session = onnxruntime.InferenceSession(path, providers=['CPUExecutionProvider']) # onnxruntime.get_available_providers())
        # Get model info
        self.get_input_details()
        self.get_output_details()
    
    @abstractmethod
    def preprocess(self, image):
        pass

    @abstractmethod
    def inference(self, input_tensor):
        pass
    
    @abstractmethod
    def run(self, input_image):
        pass

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]
