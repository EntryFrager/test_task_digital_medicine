import numpy as np
import torch

from torch.autograd import Variable
from tqdm import tqdm
from Modules import Module

    
class Linear(Module):
    """
    A module which applies a linear transformation
    A common name is fully-connected layer, InnerProductLayer in caffe.

    The module should work with 2D input of shape (n_samples, n_feature).
    """
    def __init__(self, n_in, n_out):
        super(Linear, self).__init__()

        stdv   = 1. / np.sqrt(n_in)
        self.W = np.random.uniform(-stdv, stdv, size = (n_out, n_in))
        self.b = np.random.uniform(-stdv, stdv, size = n_out)

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)


    def updateOutput(self, input):
        self.output = np.dot(input, self.W.T) + self.b

        return self.output


    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.dot(gradOutput, self.W)
        
        return self.gradInput


    def accGradParameters(self, input, gradOutput):
        self.gradW += np.dot(gradOutput.T, input)
        self.gradb += np.sum(gradOutput, axis = 0)
        
        pass


    def zeroGradParameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)


    def getParameters(self):
        return [self.W, self.b]


    def getGradParameters(self):
        return [self.gradW, self.gradb]


    def setParameters(self, parameters):
        self.W = parameters[0]
        self.b = parameters[1]


    def __repr__(self):
        s = self.W.shape
        q = 'Linear %d -> %d' %(s[1],s[0])
        return q
    

def test_Linear():
    batch_size, n_in, n_out = 2, 3, 4
    np.random.seed(42)
    torch.manual_seed(42)
    for _ in tqdm(range(100), desc="Testing Linear layer"):
        # layers initialization
        torch_layer = torch.nn.Linear(n_in, n_out)
        custom_layer = Linear(n_in, n_out)
        custom_layer.W = torch_layer.weight.data.numpy()
        custom_layer.b = torch_layer.bias.data.numpy()

        layer_input = np.random.uniform(-10, 10, (batch_size, n_in)).astype(np.float32)
        next_layer_grad = np.random.uniform(-10, 10, (batch_size, n_out)).astype(np.float32)
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

        # 3. check layer parameters grad
        custom_layer.accGradParameters(layer_input, next_layer_grad)
        weight_grad = custom_layer.gradW
        bias_grad = custom_layer.gradb
        torch_weight_grad = torch_layer.weight.grad.data.numpy()
        torch_bias_grad = torch_layer.bias.grad.data.numpy()
        np.testing.assert_allclose(
            torch_weight_grad, weight_grad, atol=3e-6,
            err_msg="Mismatch in weight gradients."
        )
        np.testing.assert_allclose(
            torch_bias_grad, bias_grad, atol=1e-6,
            err_msg="Mismatch in bias gradients."
        )


    print("\nAll tests passed successfully!")


test_Linear()