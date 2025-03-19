import numpy as np
import torch

from torch.autograd import Variable
from tqdm import tqdm
from Modules import Module
from BatchNormalization import BatchNormalization, ChannelwiseScaling


class Sequential(Module):
    """
         This class implements a container, which processes `input` data sequentially.

         `input` is processed by each module (layer) in self.modules consecutively.
         The resulting array is called `output`.
    """

    def __init__ (self):
        super(Sequential, self).__init__()
        self.modules    = []
        self.curOutput = []

    def add(self, module):
        """
        Adds a module to the container.
        """
        self.modules.append(module)

    def updateOutput(self, input):
        """
        Basic workflow of FORWARD PASS:

            y_0    = module[0].forward(input)
            y_1    = module[1].forward(y_0)
            ...
            output = module[n-1].forward(y_{n-2})


        Just write a little loop.
        """

        self.output = input

        for module in self.modules:
            self.curOutput.append(self.output)
            self.output = module.forward(self.output)

        return self.output

    def backward(self, input, gradOutput):
        """
        Workflow of BACKWARD PASS:

            g_{n-1} = module[n-1].backward(y_{n-2}, gradOutput)
            g_{n-2} = module[n-2].backward(y_{n-3}, g_{n-1})
            ...
            g_1 = module[1].backward(y_0, g_2)
            gradInput = module[0].backward(input, g_1)


        !!!

        To ech module you need to provide the input, module saw while forward pass,
        it is used while computing gradients.
        Make sure that the input for `i-th` layer the output of `module[i]` (just the same input as in forward pass)
        and NOT `input` to this Sequential module.

        !!!

        """
        
        curGradInput = gradOutput

        for i in reversed(range(len(self.modules))):
            self.gradInput = self.modules[i].backward(self.curOutput[i], curGradInput)
            curGradInput = self.gradInput

        return self.gradInput


    def zeroGradParameters(self):
        for module in self.modules:
            module.zeroGradParameters()

    def getGradParameters(self):
        """
        Should gather all gradients w.r.t parameters in a list.
        """
        return [x.getGradParameters() for x in self.modules]

    def setParameters(self, parameters):
        for x, parameter in zip(self.modules, parameters):
            x.setParameters(parameter)

    def __repr__(self):
        string = "".join([str(x) + '\n' for x in self.modules])
        return string

    def __getitem__(self,x):
        return self.modules.__getitem__(x)

    def train(self):
        """
        Propagates training parameter through all modules
        """
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        """
        Propagates training parameter through all modules
        """
        self.training = False
        for module in self.modules:
            module.evaluate()


def test_Sequential():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size, n_in = 2, 4
    for _ in tqdm(range(100), desc="Testing Sequential layer"):
        # layers initialization
        alpha = 0.9
        torch_layer = torch.nn.BatchNorm1d(n_in, eps=BatchNormalization.EPS, momentum=1.-alpha, affine=True)
        torch_layer.bias.data = torch.from_numpy(np.random.random(n_in).astype(np.float32))
        custom_layer = Sequential()
        bn_layer = BatchNormalization(alpha)
        bn_layer.moving_mean = torch_layer.running_mean.numpy().copy()
        bn_layer.moving_variance = torch_layer.running_var.numpy().copy()
        custom_layer.add(bn_layer)
        scaling_layer = ChannelwiseScaling(n_in)
        scaling_layer.gamma = torch_layer.weight.data.numpy()
        scaling_layer.beta = torch_layer.bias.data.numpy()
        custom_layer.add(scaling_layer)
        custom_layer.train()

        layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(np.float32)
        next_layer_grad = np.random.uniform(-5, 5, (batch_size, n_in)).astype(np.float32)

        # 1. check layer output
        custom_layer_output = custom_layer.updateOutput(layer_input)
        layer_input_var = Variable(torch.from_numpy(layer_input), requires_grad=True)
        torch_layer_output_var = torch_layer(layer_input_var)
        np.testing.assert_allclose(
            torch_layer_output_var.data.numpy(),
            custom_layer_output,
            atol=4e-6,
            err_msg="Mismatch in forward output between torch and custom layer."
        )

        # 2. check layer input grad
        custom_layer_grad = custom_layer.backward(layer_input, next_layer_grad)
        torch_layer_output_var.backward(torch.from_numpy(next_layer_grad))
        torch_layer_grad_var = layer_input_var.grad
        np.testing.assert_allclose(
            torch_layer_grad_var.data.numpy(),
            custom_layer_grad,
            atol=2e-4,
            err_msg="Mismatch in gradient wrt input between torch and custom layer."
        )

        # 3. check layer parameters grad
        weight_grad, bias_grad = custom_layer.getGradParameters()[1]
        torch_weight_grad = torch_layer.weight.grad.data.numpy()
        torch_bias_grad = torch_layer.bias.grad.data.numpy()
        np.testing.assert_allclose(
            torch_weight_grad, weight_grad, atol=2e-6,
            err_msg="Mismatch in weight gradients."
        )
        np.testing.assert_allclose(
            torch_bias_grad, bias_grad, atol=1e-6,
            err_msg="Mismatch in bias gradients."
        )

    print("\nAll tests passed successfully!")


test_Sequential()