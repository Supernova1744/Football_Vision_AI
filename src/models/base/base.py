import onnxruntime
from abc import ABC, abstractmethod
from .serviceProvider import serviceProvider


class BaseONNXModel(ABC):
    def __init__(self, path, providers: serviceProvider = serviceProvider.CPUExecutionProvider):
        self.initialize_model(path, providers)

    def __call__(self, image):
        return self.run(image)

    def initialize_model(self, path, providers : serviceProvider = serviceProvider.CPUExecutionProvider):
        self.session = onnxruntime.InferenceSession(path, providers=providers)
        self.get_input_details()
        self.get_output_details()
    
    @abstractmethod
    def preprocess(self, image):
        pass

    def inference(self, input_tensor):
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs
    
    @abstractmethod
    def postprocess(self, output_tensor):
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
