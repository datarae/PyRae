import numpy as np

class SGD:
    """Stochastic Gradient Descent (SGD) with optional momentum.
    
    Args:
        model (Sequential): Your initialized network stored in a `Sequential` object.
        lr (float): Learning rate, e.g., 0.01
        momentum (float, optional): Momentum factor, e.g., 0.9 (default is 0 for standard SGD).
    """
    
    def __init__(self, model, lr, momentum=0):
        self.layers = model.layers
        self.lr = lr
        self.momentum = momentum
        
        # Initialize velocity for layers with weights
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.velocity_weight = np.zeros_like(layer.weight)
                layer.velocity_bias = np.zeros_like(layer.bias)

    def zero_grad(self):
        """Resets the gradients of weights to zero."""
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.grad_weight.fill(0)
                layer.grad_bias.fill(0)

    def step(self):
        """Updates the weights and biases using SGD with momentum."""
        for layer in self.layers:
            if isinstance(layer, Linear):
                # Update velocity for weights and biases
                layer.velocity_weight = self.momentum * layer.velocity_weight + (1 - self.momentum) * layer.grad_weight
                layer.velocity_bias = self.momentum * layer.velocity_bias + (1 - self.momentum) * layer.grad_bias

                # Update weights and biases using velocity
                layer.weight -= self.lr * layer.velocity_weight
                layer.bias -= self.lr * layer.velocity_bias


class Adam:
    """Adam optimizer with adaptive learning rates for each parameter."""
    
    def __init__(self, model, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.layers = model.layers
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # Timestep for bias correction
        
        # Initialize moment estimates
        for layer in self.layers:
            if isinstance(layer, Linear):
                layer.m = np.zeros_like(layer.weight)  # First moment (mean)
                layer.v = np.zeros_like(layer.weight)  # Second moment (variance)
                layer.m_bias = np.zeros_like(layer.bias)
                layer.v_bias = np.zeros_like(layer.bias)

    def step(self):
        self.t += 1
        for layer in self.layers:
            if isinstance(layer, Linear):
                # Update first and second moments
                layer.m = self.beta1 * layer.m + (1 - self.beta1) * layer.grad_weight
                layer.v = self.beta2 * layer.v + (1 - self.beta2) * (layer.grad_weight ** 2)
                
                # Bias correction
                m_hat = layer.m / (1 - self.beta1 ** self.t)
                v_hat = layer.v / (1 - self.beta2 ** self.t)

                # Update parameters
                layer.weight -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

                # Repeat for bias
                layer.m_bias = self.beta1 * layer.m_bias + (1 - self.beta1) * layer.grad_bias
                layer.v_bias = self.beta2 * layer.v_bias + (1 - self.beta2) * (layer.grad_bias ** 2)
                
                m_hat_bias = layer.m_bias / (1 - self.beta1 ** self.t)
                v_hat_bias = layer.v_bias / (1 - self.beta2 ** self.t)

                layer.bias -= self.lr * m_hat_bias / (np.sqrt(v_hat_bias) + self.eps)
