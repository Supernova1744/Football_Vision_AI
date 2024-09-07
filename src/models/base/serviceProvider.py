import onnxruntime
from dataclasses import dataclass

@dataclass
class serviceProvider:
    TensorrtExecutionProvider = ["TensorrtExecutionProvider"]
    CUDAExecutionProvider = ["CUDAExecutionProvider"]
    CPUExecutionProvider = ["CPUExecutionProvider"]
    ALL = onnxruntime.get_available_providers()