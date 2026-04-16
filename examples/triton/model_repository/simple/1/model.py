import triton_python_backend_utils as pb_utils
import numpy as np
import random
import time

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            # Небольшая рандомная задержка (от 5 до 25 мс)
            time.sleep(random.uniform(0.005, 0.025))
            
            # Имитация ошибки в 1% случаев
            if random.random() < 0.01:
                err_msg = "Synthetic random error for testing"
                # Triton требует специфичный тип ошибки. Попробуем найти его.
                error_obj = None
                if hasattr(pb_utils, "TritonModelError"):
                    error_obj = pb_utils.TritonModelError(err_msg)
                elif hasattr(pb_utils, "TritonError"):
                    error_obj = pb_utils.TritonError(err_msg)
                
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], 
                    error=error_obj
                ))
                continue

            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            if in_0 is None:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], 
                    error=pb_utils.TritonModelError("INPUT0 not found") if hasattr(pb_utils, "TritonModelError") else None
                ))
                continue

            out_tensor_0 = pb_utils.Tensor("OUTPUT0", in_0.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))
        return responses
