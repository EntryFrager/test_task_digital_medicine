import numpy as np
import torch

from torch.autograd import Variable
from tqdm import tqdm
from Modules.BasicModule import Module


class MaxPool2d(Module):
    def __init__(self, kernel_size):
        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.gradInput = None

    def updateOutput(self, input):
        input_h, input_w = input.shape[-2:]
        # your may remove these asserts and implement MaxPool2d with padding
        assert input_h % self.kernel_size == 0
        assert input_w % self.kernel_size == 0

        # YOUR CODE #############################
        # self.output = ...
        # self.max_indices = ...
        return self.output

    def updateGradInput(self, input, gradOutput):
        # YOUR CODE #############################
        # self.gradInput = ...
        return self.gradInput

    def __repr__(self):
        q = 'MaxPool2d, kern %d, stride %d' %(self.kernel_size, self.kernel_size)
        return q
    

def test_MaxPool2d():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size, n_in = 2, 3
    h,w = 4,6
    kern_size = 2
    for _ in tqdm(range(100), desc="Testing MaxPool2d layer"):
        # layers initialization
        torch_layer = torch.nn.MaxPool2d(kern_size)
        custom_layer = MaxPool2d(kern_size)

        layer_input = np.random.uniform(-10, 10, (batch_size, n_in, h,w)).astype(np.float32)
        next_layer_grad = np.random.uniform(-10, 10, (batch_size, n_in,
                                                        h // kern_size, w // kern_size)).astype(np.float32)

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


test_MaxPool2d()