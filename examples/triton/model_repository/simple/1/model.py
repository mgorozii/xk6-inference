import triton_python_backend_utils as pb_utils
import numpy as np
import random
import time

class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            time.sleep(random.uniform(0.005, 0.025))
            
            if random.random() < 0.01:
                err_msg = "Synthetic random error for testing"
                
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], 
                    error=pb_utils.TritonError(err_msg)
                ))
                continue

            in_0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            if in_0 is None:
                responses.append(pb_utils.InferenceResponse(
                    output_tensors=[], 
                    error=pb_utils.TritonError("INPUT0 not found")
                ))
                continue

            out_tensor_0 = pb_utils.Tensor("OUTPUT0", in_0.as_numpy())
            responses.append(pb_utils.InferenceResponse([out_tensor_0]))
        return responses
