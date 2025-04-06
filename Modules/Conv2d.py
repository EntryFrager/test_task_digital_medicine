import numpy as np
import torch
import scipy as sp

from torch.autograd import Variable
from tqdm import tqdm
from Modules.BasicModule import Module


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Conv2d, self).__init__()
        assert kernel_size % 2 == 1, kernel_size

        stdv = 1./np.sqrt(in_channels)
        self.W = np.random.uniform(-stdv, stdv, size = (out_channels, in_channels, kernel_size, kernel_size))
        self.b = np.random.uniform(-stdv, stdv, size = (out_channels,))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)


    def updateOutput(self, input):
        pad_size = self.kernel_size // 2

        input_pad = np.pad(input, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)))
        batch_size, _, height, width = input.shape
        
        self.output = np.zeros((batch_size, self.out_channels, height, width))

        for n_batch in range(batch_size):
            for out_channel in range(self.out_channels):
                channel_sum = np.zeros((height, width))
                
                for in_channel in range(self.in_channels):
                    cur_input = input_pad[n_batch, in_channel]
                    kernel = self.W[out_channel, in_channel]

                    channel_sum += sp.signal.correlate(cur_input, kernel, mode = 'valid')

                self.output[n_batch, out_channel] = channel_sum + self.b[out_channel]

        return self.output


    def updateGradInput(self, input, gradOutput):
        pad_size = self.kernel_size // 2

        gradOutputPad = np.pad(gradOutput, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)))
 
        batch_size, _, height, width = input.shape

        self.gradInput = np.zeros_like(input)       
        
        for n_batch in range(batch_size):
            for in_channel in range(self.in_channels):
                grad = np.zeros((height, width))
                for out_channel in range(self.out_channels):
                    cur_grad = gradOutputPad[n_batch, out_channel]
                    kernel = self.W[out_channel, in_channel][::-1, ::-1]

                    grad += sp.signal.correlate(cur_grad, kernel, mode = 'valid')

                self.gradInput[n_batch, in_channel] = grad

        return self.gradInput


    def accGradParameters(self, input, gradOutput):
        pad_size = self.kernel_size // 2

        input_padded = np.pad(input, ((0, 0), (0, 0), (pad_size, pad_size), (pad_size, pad_size)))
        batch_size = input.shape[0]
        
        for n_batch in range(batch_size):
            for out_channel in range(self.out_channels):
                grad = gradOutput[n_batch, out_channel]

                for in_channel in range(self.in_channels):
                    cur_input = input_padded[n_batch, in_channel]
                    
                    self.gradW[out_channel, in_channel] += sp.signal.correlate(cur_input, grad, mode = 'valid')
        
        self.gradb += np.sum(gradOutput, axis = (0, 2, 3))


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
        q = 'Conv2d %d -> %d' %(s[1],s[0])
        return q
    

def test_Conv2d():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size, n_in, n_out = 2, 3, 4
    h,w = 5,6
    kern_size = 3
    for _ in tqdm(range(100), desc="Testing Conv2d layer"):
        # layers initialization
        torch_layer = torch.nn.Conv2d(n_in, n_out, kern_size, padding=1)
        custom_layer = Conv2d(n_in, n_out, kern_size)
        custom_layer.W = torch_layer.weight.data.numpy() # [n_out, n_in, kern, kern]
        custom_layer.b = torch_layer.bias.data.numpy()

        layer_input = np.random.uniform(-1, 1, (batch_size, n_in, h,w)).astype(np.float32)
        next_layer_grad = np.random.uniform(-1, 1, (batch_size, n_out, h, w)).astype(np.float32)

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
        #m = ~np.isclose(torch_weight_grad, weight_grad, atol=1e-5)
        np.testing.assert_allclose(
            torch_weight_grad, weight_grad, atol=4e-6,
            err_msg="Mismatch in weight gradients."
        )
        np.testing.assert_allclose(
            torch_bias_grad, bias_grad, atol=4e-6,
            err_msg="Mismatch in bias gradients."
        )

    print("\nAll tests passed successfully!")


# test_Conv2d()