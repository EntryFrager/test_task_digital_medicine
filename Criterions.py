import numpy as np
import torch

from torch.autograd import Variable
from tqdm import tqdm


class Criterion(object):
    def __init__ (self):
        self.output = None
        self.gradInput = None


    def forward(self, input, target):
        """
            Given an input and a target, compute the loss function
            associated to the criterion and return the result.

            For consistency this function should not be overrided,
            all the code goes in `updateOutput`.
        """
        return self.updateOutput(input, target)


    def backward(self, input, target):
        """
            Given an input and a target, compute the gradients of the loss function
            associated to the criterion and return the result.

            For consistency this function should not be overrided,
            all the code goes in `updateGradInput`.
        """
        return self.updateGradInput(input, target)


    def updateOutput(self, input, target):
        """
        Function to override.
        """
        return self.output


    def updateGradInput(self, input, target):
        """
        Function to override.
        """
        return self.gradInput


    def __repr__(self):
        """
        Pretty printing. Should be overrided in every module if you want
        to have readable description.
        """
        return "Criterion"
    

class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()


    def updateOutput(self, input, target):
        self.output = np.sum(np.power(input - target, 2)) / input.shape[0]
        return self.output


    def updateGradInput(self, input, target):
        self.gradInput  = (input - target) * 2 / input.shape[0]
        return self.gradInput


    def __repr__(self):
        return "MSECriterion"
    

class ClassNLLCriterionUnstable(Criterion):
    EPS = 1e-15


    def __init__(self):
        a = super(ClassNLLCriterionUnstable, self)
        super(ClassNLLCriterionUnstable, self).__init__()


    def updateOutput(self, input, target):
        input_clamp = np.clip(input, self.EPS, 1 - self.EPS)
        self.output = - np.sum(target * np.log(input_clamp)) / input.shape[0]
        
        return self.output


    def updateGradInput(self, input, target):
        input_clamp    = np.clip(input, self.EPS, 1 - self.EPS)
        self.gradInput = - target / (input_clamp * input.shape[0])  

        return self.gradInput


    def __repr__(self):
        return "ClassNLLCriterionUnstable"
    

class ClassNLLCriterion(Criterion):
    def __init__(self):
        a = super(ClassNLLCriterion, self)
        super(ClassNLLCriterion, self).__init__()


    def updateOutput(self, input, target):
        self.output = - np.sum(target * input) / input.shape[0]
        return self.output


    def updateGradInput(self, input, target):
        self.gradInput = - target / input.shape[0]
        return self.gradInput


    def __repr__(self):
        return "ClassNLLCriterion"
    

def test_NLLCriterionUnstable():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size, n_in = 2, 4
    for _ in tqdm(range(100), desc="Testing NLLCriterionUnstable layer"):
        # layers initialization
        torch_layer = torch.nn.NLLLoss()
        custom_layer = ClassNLLCriterionUnstable()

        layer_input = np.random.uniform(0, 1, (batch_size, n_in)).astype(np.float32)
        layer_input /= layer_input.sum(axis=-1, keepdims=True)
        layer_input = layer_input.clip(custom_layer.EPS, 1. - custom_layer.EPS)  # unifies input
        target_labels = np.random.choice(n_in, batch_size)
        target = np.zeros((batch_size, n_in), np.float32)
        target[np.arange(batch_size), target_labels] = 1  # one-hot encoding

        # 1. check layer output
        custom_layer_output = custom_layer.updateOutput(layer_input, target)
        layer_input_var = Variable(torch.from_numpy(layer_input), requires_grad=True)
        torch_layer_output_var = torch_layer(torch.log(layer_input_var),
                                                Variable(torch.from_numpy(target_labels), requires_grad=False))
        np.testing.assert_allclose(
            torch_layer_output_var.data.numpy(),
            custom_layer_output,
            atol=1e-6,
            err_msg="Mismatch in forward output between torch and custom layer."
        )

        # 2. check layer input grad
        custom_layer_grad = custom_layer.updateGradInput(layer_input, target)
        torch_layer_output_var.backward()
        torch_layer_grad_var = layer_input_var.grad
        np.testing.assert_allclose(
            torch_layer_grad_var.data.numpy(),
            custom_layer_grad,
            atol=1e-6,
            err_msg="Mismatch in gradient wrt input between torch and custom layer."
        )

    print("\nAll tests passed successfully!")


def test_NLLCriterion():
    np.random.seed(42)
    torch.manual_seed(42)

    batch_size, n_in = 2, 4
    for _ in tqdm(range(100), desc="Testing NLLCriterion layer"):
        # layers initialization
        torch_layer = torch.nn.NLLLoss()
        custom_layer = ClassNLLCriterion()

        layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(np.float32)
        layer_input = torch.nn.LogSoftmax(dim=1)(Variable(torch.from_numpy(layer_input))).data.numpy()
        target_labels = np.random.choice(n_in, batch_size)
        target = np.zeros((batch_size, n_in), np.float32)
        target[np.arange(batch_size), target_labels] = 1  # one-hot encoding

        # 1. check layer output
        custom_layer_output = custom_layer.updateOutput(layer_input, target)
        layer_input_var = Variable(torch.from_numpy(layer_input), requires_grad=True)
        torch_layer_output_var = torch_layer(layer_input_var,
                                                Variable(torch.from_numpy(target_labels), requires_grad=False))
        np.testing.assert_allclose(
            torch_layer_output_var.data.numpy(),
            custom_layer_output,
            atol=1e-6,
            err_msg="Mismatch in forward output between torch and custom layer."
        )

        # 2. check layer input grad
        custom_layer_grad = custom_layer.updateGradInput(layer_input, target)
        torch_layer_output_var.backward()
        torch_layer_grad_var = layer_input_var.grad
        np.testing.assert_allclose(
            torch_layer_grad_var.data.numpy(),
            custom_layer_grad,
            atol=1e-6,
            err_msg="Mismatch in gradient wrt input between torch and custom layer."
        )

    print("\nAll tests passed successfully!")


# test_NLLCriterionUnstable()

# test_NLLCriterion()