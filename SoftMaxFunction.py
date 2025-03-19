import numpy as np
import torch

from torch.autograd import Variable
from tqdm import tqdm
from Modules import Module

class SoftMax(Module):
    def __init__(self):
         super(SoftMax, self).__init__()


    def updateOutput(self, input):
        # start with normalization for numerical stability
        norm_input  = np.subtract(input, input.max(axis = 1, keepdims = True))
        exp_norm    = np.exp(norm_input)
        self.output = exp_norm / np.sum(exp_norm, axis = 1, keepdims = True)

        return self.output


    def updateGradInput(self, input, gradOutput):
        sum_grad       = np.sum(gradOutput * self.output, axis = 1, keepdims = True)
        self.gradInput = self.output * (gradOutput - sum_grad)

        return self.gradInput


    def __repr__(self):
        return "SoftMax"
    

def test_SoftMax():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size, n_in = 2, 4
    for _ in tqdm(range(100), desc="Testing SoftMax layer"):
        # layers initialization
        torch_layer = torch.nn.Softmax(dim=1)
        custom_layer = SoftMax()

        layer_input = np.random.uniform(-10, 10, (batch_size, n_in)).astype(np.float32)
        next_layer_grad = np.random.random((batch_size, n_in)).astype(np.float32)
        next_layer_grad /= next_layer_grad.sum(axis=-1, keepdims=True)
        next_layer_grad = next_layer_grad.clip(1e-5,1.)
        next_layer_grad = 1. / next_layer_grad

        # 1. check layer output
        custom_layer_output = custom_layer.updateOutput(layer_input)
        layer_input_var = Variable(torch.from_numpy(layer_input), requires_grad=True)
        torch_layer_output_var = torch_layer(layer_input_var)
        np.testing.assert_allclose(
            torch_layer_output_var.data.numpy(),
            custom_layer_output,
            atol=1e-5,
            err_msg="Mismatch in forward output between torch and custom layer."
        )

        # 2. check layer input grad
        custom_layer_grad = custom_layer.updateGradInput(layer_input, next_layer_grad)
        torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
        torch_layer_grad_var = layer_input_var.grad
        np.testing.assert_allclose(
            torch_layer_grad_var.data.numpy(),
            custom_layer_grad,
            atol=1.5e-5,
            err_msg="Mismatch in gradient wrt input between torch and custom layer."
        )

    print("\nAll tests passed successfully!")


test_SoftMax()