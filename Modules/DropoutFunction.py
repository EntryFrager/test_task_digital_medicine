import numpy as np

from tqdm import tqdm
from BasicModule import Module


class Dropout(Module):
    def __init__(self, p = 0.5):
        super(Dropout, self).__init__()

        self.p = p
        self.mask = None


    def updateOutput(self, input):
        if self.training:
            self.mask = (np.random.rand(*input.shape) > self.p).astype(np.float32)
            self.output = input * self.mask / (1 - self.p)
        else:
            self.output = input.copy()

        return  self.output


    def updateGradInput(self, input, gradOutput):
        if self.training:
            self.gradInput = gradOutput * self.mask / (1 - self.p)
        else:
            self.gradInput = gradOutput.copy()
            
        return self.gradInput


    def getParameters(self):
        return [self.p, self.mask]


    def setParameters(self, parameters):
        self.p = parameters[0]
        self.mask = parameters[1]


    def __repr__(self):
        return "Dropout"
    

def test_Dropout():
    np.random.seed(42)
    batch_size, n_in = 2, 4

    for _ in tqdm(range(100), desc="Testing Dropout layer"):
        # layers initialization
        p = np.random.uniform(0.3, 0.7)
        layer = Dropout(p)
        layer.train()

        layer_input = np.random.uniform(-5, 5, (batch_size, n_in)).astype(np.float32)
        next_layer_grad = np.random.uniform(-5, 5, (batch_size, n_in)).astype(np.float32)

        # 1. check layer output
        layer_output = layer.updateOutput(layer_input)
        mask = np.logical_or(np.isclose(layer_output, 0),
                             np.isclose(layer_output * (1. - p), layer_input))
        np.testing.assert_array_equal(mask, True), "Mismatch in dropout layer output."

        # 2. check layer input grad
        layer_grad = layer.updateGradInput(layer_input, next_layer_grad)
        mask_grad = np.logical_or(np.isclose(layer_grad, 0),
                                  np.isclose(layer_grad * (1. - p), next_layer_grad))
        np.testing.assert_array_equal(mask_grad, True), "Mismatch in dropout layer grad input."

        # 3. check evaluation mode
        layer.evaluate()
        layer_output = layer.updateOutput(layer_input)
        np.testing.assert_allclose(layer_output, layer_input, atol=1e-6), "Dropout evaluation mode failed."


        # 4. check mask
        p = 0.0
        layer = Dropout(p)
        layer.train()
        layer_output = layer.updateOutput(layer_input)
        np.testing.assert_allclose(layer_output, layer_input, atol=1e-6), "Dropout with p=0 should return input unchanged."

        p = 0.5
        layer = Dropout(p)
        layer.train()
        layer_input = np.random.uniform(5, 10, (batch_size, n_in)).astype(np.float32)
        next_layer_grad = np.random.uniform(5, 10, (batch_size, n_in)).astype(np.float32)
        layer_output = layer.updateOutput(layer_input)
        zeroed_elem_mask = np.isclose(layer_output, 0)
        layer_grad = layer.updateGradInput(layer_input, next_layer_grad)
        np.testing.assert_array_equal(zeroed_elem_mask, np.isclose(layer_grad, 0)), "Mismatch in dropout mask."

        # 5. dropout mask should be generated independently for every input matrix element, not for row/column
        batch_size_large, n_in_large = 1000, 1
        p = 0.8
        layer = Dropout(p)
        layer.train()

        layer_input = np.random.uniform(5, 10, (batch_size_large, n_in_large)).astype(np.float32)
        layer_output = layer.updateOutput(layer_input)
        assert np.sum(np.isclose(layer_output, 0)) != layer_input.size, "Dropout mask should be applied independently."

        # Test with transposed input
        layer_input = layer_input.T
        layer_output = layer.updateOutput(layer_input)
        assert np.sum(np.isclose(layer_output, 0)) != layer_input.size, "Dropout mask should be applied independently for transposed input."

    print("\nAll tests passed successfully!")


# test_Dropout()