import numpy as np
import torch

from torch.autograd import Variable
from tqdm import tqdm
from Modules import Module


class ReLU(Module):
    def __init__(self):
         super(ReLU, self).__init__()


    def updateOutput(self, input):
        self.output = np.maximum(input, 0)
        return self.output


    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.multiply(gradOutput , input > 0)
        return self.gradInput


    def __repr__(self):
        return "ReLU"
    

class LeakyReLU(Module):
    def __init__(self, slope = 0.03):
        super(LeakyReLU, self).__init__()

        self.slope = slope


    def updateOutput(self, input):
        self.output = np.where(input > 0, input, input * self.slope)
        return  self.output


    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(input > 0, 1.0, self.slope) * gradOutput
        return self.gradInput


    def __repr__(self):
        return "LeakyReLU"


class ELU(Module):
    def __init__(self, alpha = 1.0):
        super(ELU, self).__init__()

        self.alpha = alpha


    def updateOutput(self, input):
        self.output = np.where(input > 0, input, self.alpha * (np.exp(input) - 1))
        return  self.output


    def updateGradInput(self, input, gradOutput):
        self.gradInput = np.where(input > 0, 1.0, self.alpha * np.exp(input)) * gradOutput
        return self.gradInput


    def __repr__(self):
        return "ELU"
    

class SoftPlus(Module):
    def __init__(self):
        super(SoftPlus, self).__init__()


    def updateOutput(self, input):
        self.output = np.log(1 + np.exp(input))
        return  self.output


    def updateGradInput(self, input, gradOutput):
        self.gradInput = gradOutput * np.exp(input) / (1 + np.exp(input))
        return self.gradInput


    def __repr__(self):
        return "SoftPlus"
    

def test_LeakyReLU():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size, n_in = 2, 4
    for _ in tqdm(range(100), desc="Testing LeakyReLU layer"):
        # layers initialization
        slope = np.random.uniform(0.01, 0.05)
        torch_layer = torch.nn.LeakyReLU(slope)
        custom_layer = LeakyReLU(slope)

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
        np.testing.assert_allclose(
            torch_layer_grad_var.data.numpy(),
            custom_layer_grad,
            atol=1e-6,
            err_msg="Mismatch in gradient wrt input between torch and custom layer."
        )

    print("\nAll tests passed successfully!")


def test_ELU():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size, n_in = 2, 4
    for _ in tqdm(range(100), desc="Testing ELU layer"):
        # layers initialization
        alpha = 1.0
        torch_layer = torch.nn.ELU(alpha)
        custom_layer = ELU(alpha)

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
        np.testing.assert_allclose(
            torch_layer_grad_var.data.numpy(),
            custom_layer_grad,
            atol=1e-6,
            err_msg="Mismatch in gradient wrt input between torch and custom layer."
        )

    print("\nAll tests passed successfully!")


def test_SoftPlus():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size, n_in = 2, 4
    for _ in tqdm(range(100), desc="Testing SoftPlus layer"):
        # layers initialization
        torch_layer = torch.nn.Softplus()
        custom_layer = SoftPlus()

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
        np.testing.assert_allclose(
            torch_layer_grad_var.data.numpy(),
            custom_layer_grad,
            atol=1e-6,
            err_msg="Mismatch in gradient wrt input between torch and custom layer."
        )

    print("\nAll tests passed successfully!")


test_LeakyReLU()

test_ELU()

test_SoftPlus()