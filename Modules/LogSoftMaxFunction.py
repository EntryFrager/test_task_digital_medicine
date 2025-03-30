import numpy as np
import torch

from torch.autograd import Variable
from tqdm import tqdm
from Modules.BasicModule import Module


class LogSoftMax(Module):
    def __init__(self):
        super(LogSoftMax, self).__init__()
        self.softmax = None


    def updateOutput(self, input):
        norm_input     = np.subtract(input, input.max(axis = 1, keepdims = True))
        exp_input      = np.exp(norm_input)
        sum_exp_offset = np.sum(exp_input, axis = 1, keepdims = True)
        self.softmax   = exp_input / sum_exp_offset
        self.output    = norm_input - np.log(sum_exp_offset)

        return self.output


    def updateGradInput(self, input, gradOutput):
        sum_grad       = np.sum(gradOutput, axis = 1, keepdims = True)
        self.gradInput = gradOutput - self.softmax * sum_grad

        return self.gradInput


    def __repr__(self):
        return "LogSoftMax"
    

def test_LogSoftMax():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size, n_in = 2, 4
    for _ in tqdm(range(100), desc="Testing LogSoftMax layer"):
        # layers initialization
        torch_layer = torch.nn.LogSoftmax(dim=1)
        custom_layer = LogSoftMax()

        layer_input = np.random.uniform(-10, 10, (batch_size, n_in)).astype(np.float32)
        next_layer_grad = np.random.random((batch_size, n_in)).astype(np.float32)
        next_layer_grad /= next_layer_grad.sum(axis=-1, keepdims=True)

        # 1. check layer output
        custom_layer_output = custom_layer.updateOutput(layer_input)
        layer_input_var = Variable(torch.from_numpy(layer_input), requires_grad=True)
        torch_layer_output_var = torch_layer(layer_input_var)
        np.testing.assert_allclose(
            torch_layer_output_var.data.numpy(),
            custom_layer_output,
            atol=1e-6,
            err_msg="Mismatch in forward output between torch and custom layer."
        )
        # 2. check layer input grad
        custom_layer_grad = custom_layer.updateGradInput(layer_input, next_layer_grad)
        torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
        torch_layer_grad_var = layer_input_var.grad
        np.testing.assert_allclose(
            torch_layer_grad_var.data.numpy(),
            custom_layer_grad,
            atol=1e-6,
            err_msg="Mismatch in gradient wrt input between torch and custom layer."
        )

    print("\nAll tests passed successfully!")


# test_LogSoftMax()