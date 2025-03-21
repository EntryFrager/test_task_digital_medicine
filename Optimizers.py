import numpy as np

def sgd_momentum(variables, gradients, config, state):
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('accumulated_grads', {})

    var_index = 0
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):

            old_grad = state['accumulated_grads'].setdefault(var_index, np.zeros_like(current_grad))

            np.add(config['momentum'] * old_grad, config['learning_rate'] * current_grad, out=old_grad)

            current_var -= old_grad
            var_index += 1

        
def adam_optimizer(variables, gradients, config, state):
    # 'variables' and 'gradients' have complex structure, accumulated_grads will be stored in a simpler one
    state.setdefault('m', {})  # first moment vars
    state.setdefault('v', {})  # second moment vars
    state.setdefault('t', 0)   # timestamp
    state['t'] += 1
    for k in ['learning_rate', 'beta1', 'beta2', 'epsilon']:
        assert k in config, config.keys()

    var_index = 0
    lr_t = config['learning_rate'] * np.sqrt(1 - config['beta2']**state['t']) / (1 - config['beta1']**state['t'])
    for current_layer_vars, current_layer_grads in zip(variables, gradients):
        for current_var, current_grad in zip(current_layer_vars, current_layer_grads):
            var_first_moment = state['m'].setdefault(var_index, np.zeros_like(current_grad))
            var_second_moment = state['v'].setdefault(var_index, np.zeros_like(current_grad))

            np.add(config['beta1'] * var_first_moment, (1 - config['beta1']) * current_grad, out = var_first_moment)
            np.add(config['beta2'] * var_second_moment, (1 - config['beta2']) * current_grad ** 2, out = var_second_moment)
            current_var -= lr_t * var_first_moment / (var_second_moment ** 0.5 + config['epsilon'])

            # small checks that you've updated the state; use np.add for rewriting np.arrays values
            assert var_first_moment is state['m'].get(var_index)
            assert var_second_moment is state['v'].get(var_index)
            var_index += 1


def test_adam_optimizer():
    state = {}
    config = {'learning_rate': 1e-3, 'beta1': 0.9, 'beta2':0.999, 'epsilon':1e-8}
    variables = [[np.arange(10).astype(np.float64)]]
    gradients = [[np.arange(10).astype(np.float64)]]

    # First update
    adam_optimizer(variables, gradients, config, state)
    np.testing.assert_allclose(
        state['m'][0],
        np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]),
        err_msg="Mismatch in state['m'] after first update"
    )
    np.testing.assert_allclose(
        state['v'][0],
        np.array([0.0, 0.001, 0.004, 0.009, 0.016, 0.025, 0.036, 0.049, 0.064, 0.081]),
        err_msg="Mismatch in state['v'] after first update"
    )
    np.testing.assert_equal(
        state['t'],
        1,
        err_msg="Mismatch in state['t'] after first update"
    )
    np.testing.assert_allclose(
        variables[0][0],
        np.array([0.0, 0.999, 1.999, 2.999, 3.999, 4.999, 5.999, 6.999, 7.999, 8.999]),
        err_msg="Mismatch in variables after first update"
    )

    # Second update
    adam_optimizer(variables, gradients, config, state)
    np.testing.assert_allclose(
        state['m'][0],
        np.array([0.0, 0.19, 0.38, 0.57, 0.76, 0.95, 1.14, 1.33, 1.52, 1.71]),
        err_msg="Mismatch in state['m'] after second update"
    )
    np.testing.assert_allclose(
        state['v'][0],
        np.array([0.0, 0.001999, 0.007996, 0.017991, 0.031984, 0.049975, 0.071964, 0.097951, 0.127936, 0.161919]),
        err_msg="Mismatch in state['v'] after second update"
    )
    np.testing.assert_equal(
        state['t'],
        2,
        err_msg="Mismatch in state['t'] after second update"
    )
    np.testing.assert_allclose(
        variables[0][0],
        np.array([0.0, 0.998, 1.998, 2.998, 3.998, 4.998, 5.998, 6.998, 7.998, 8.998]),
        err_msg="Mismatch in variables after second update"
    )

    print("All tests passed successfully!")


# test_adam_optimizer()