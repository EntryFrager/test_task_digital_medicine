import numpy as np
import torch

from torch.autograd import Variable
from tqdm import tqdm
from Modules import Module


class BatchNormalization(Module):
    EPS = 1e-3


    def __init__(self, alpha = 0.):
        super(BatchNormalization, self).__init__()
        self.alpha           = alpha
        self.moving_mean     = None
        self.moving_variance = None

        self.mean            = None
        self.variance        = None


    def updateOutput(self, input):
        if self.training:
            self.mean     = np.mean(input, axis = 0)
            self.variance = np.var(input, axis = 0)
            
            if self.moving_mean is None:
                self.moving_mean     = self.mean.copy()
                self.moving_variance = self.variance.copy()
            else:
                self.moving_mean     = self.alpha * self.moving_mean + self.mean * (1 - self.alpha)
                self.moving_variance = self.alpha * self.moving_variance + self.variance * (1 - self.alpha)
        else:
            if self.moving_mean is None:
                self.moving_mean     = self.mean.copy()
                self.moving_variance = self.variance.copy()

            self.mean     = self.moving_mean
            self.variance = self.moving_variance

        self.output = (input - self.mean) / np.sqrt(self.variance + self.EPS)

        return self.output
    

    def updateGradInput(self, input, gradOutput):
        batch_size = input.shape[0]
        std_inv    = 1.0 / np.sqrt(self.variance + self.EPS)

        d_norm = gradOutput
        d_var  = np.sum(d_norm * (input - self.mean) * - 0.5 * std_inv ** 3, axis = 0)
        d_mean = np.sum(d_norm * - std_inv, axis = 0) + d_var * np.mean(- 2.0 * (input - self.mean), axis = 0)

        self.gradInput = d_norm * std_inv + d_var * 2 * (input - self.mean) / batch_size + d_mean / batch_size
        return self.gradInput
    

    def getParameters(self):
        return [self.moving_mean, self.moving_variance]


    def setParameters(self, parameters):
        self.moving_mean = parameters[0]
        self.moving_variance = parameters[1]


    def __repr__(self):
        return "BatchNormalization"
    

class ChannelwiseScaling(Module):
    """
       Implements linear transform of input y = gamma * x + beta
       where gamma, beta - learnable vectors of length x.shape[-1]
    """
    def __init__(self, n_out):
        super(ChannelwiseScaling, self).__init__()

        stdv = 1./np.sqrt(n_out)
        self.gamma = np.random.uniform(-stdv, stdv, size=n_out)
        self.beta = np.random.uniform(-stdv, stdv, size=n_out)

        self.gradGamma = np.zeros_like(self.gamma)
        self.gradBeta = np.zeros_like(self.beta)


    def updateOutput(self, input):
        self.output = input * self.gamma + self.beta
        return self.output


    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * self.gamma
        return self.gradInput


    def accGradParameters(self, input, gradOutput):
        self.gradBeta = np.sum(gradOutput, axis = 0)
        self.gradGamma = np.sum(gradOutput * input, axis = 0)


    def zeroGradParameters(self):
        self.gradGamma.fill(0)
        self.gradBeta.fill(0)


    def getParameters(self):
        return [self.gamma, self.beta]


    def getGradParameters(self):
        return [self.gradGamma, self.gradBeta]


    def setParameters(self, parameters):
        self.gamma = parameters[0]
        self.beta = parameters[1]


    def __repr__(self):
        return "ChannelwiseScaling"
    

def test_BatchNormalization():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size, n_in = 32, 16
    for _ in tqdm(range(100), desc="Testing BacthNormalization layer"):
        # layers initialization
        slope = np.random.uniform(0.01, 0.05)
        alpha = 0.9
        custom_layer = BatchNormalization(alpha)
        custom_layer.train()
        torch_layer = torch.nn.BatchNorm1d(n_in, eps=custom_layer.EPS, momentum=1.-alpha, affine=False)
        custom_layer.moving_mean = torch_layer.running_mean.numpy().copy()
        custom_layer.moving_variance = torch_layer.running_var.numpy().copy()

        layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(np.float32)
        next_layer_grad = np.random.uniform(-5, 5, (batch_size, n_in)).astype(np.float32)

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
        # please, don't increase `atol` parameter, it's garanteed that you can implement batch norm layer
        # with tolerance 1e-5
        np.testing.assert_allclose(
            torch_layer_grad_var.data.numpy(),
            custom_layer_grad,
            atol=1e-5,
            err_msg="Mismatch in gradient wrt input between torch and custom layer."
        )

        # 3. check moving mean
        np.testing.assert_allclose(
            torch_layer.running_mean.numpy(),
            custom_layer.moving_mean,
            atol=1e-7,
            err_msg="Mismatch in moving mean between torch and custom layer."
        )
        # we don't check moving_variance because pytorch uses slightly different formula for it:
        # it computes moving average for unbiased variance (i.e var*N/(N-1))
        # np.testing.assert_allclose(
        #     torch_layer.running_var.numpy(),
        #     custom_layer.moving_variance,
        #     atol=1e-8,
        #     err_msg="Mismatch in moving variance between torch and custom layer."
        # )

        # 4. check evaluation mode
        custom_layer.moving_variance = torch_layer.running_var.numpy().copy()
        custom_layer.evaluate()
        custom_layer_output = custom_layer.updateOutput(layer_input)
        torch_layer.eval()
        torch_layer_output_var = torch_layer(layer_input_var)
        np.testing.assert_allclose(
            torch_layer_output_var.data.numpy(),
            custom_layer_output,
            atol=1e-6,
            err_msg="Mismatch in forward output between torch and custom layer in evaluation mode."
        )

    print("\nAll tests passed successfully!")


test_BatchNormalization()