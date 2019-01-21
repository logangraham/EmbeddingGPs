import pyro
import torch
import pyro.distributions as dist
import pyro.contrib.gp as gp
import matplotlib.pyplot as plt
from torch import nn

torch.manual_seed(42)


def generate_data(n=1000, p=3):
    """
    Generate random data where Y = sin(X)Beta.T + sigma, i.e. a linear
    sigmoidal function, where:
        X ~ U[-5, 5]
        coefs ~ DiagMVN(M_0, I) with I.shape = (p, p)
        y_i = sin(X_i) * coefs + e_i, where ei ~ Normal(0, 0.05)

    Args:
        n (int): num data points
        p (int): num dimensions
    """
    X = dist.Uniform(-5, 5).sample((n, p))
    coefs = dist.Normal(0, 1).sample((p, 1))
    y = torch.matmul(torch.sin(X), coefs) + dist.Normal(0, 0.05).sample((n, 1))
    return X, y


class HalfAutoEncoder(nn.Module):
    """
    A two-layer fully-connected network with a decreasing layer-size.

    Attributes:
        l1 (nn.modules.linear.Linear): the first layer.
        l1 (nn.modules.linear.Linear): the second layer.
    """

    def __init__(self, p_in, h_dim, p_out):
        """
        Initialize the Half Autoencoder. Assumes a two-hidden-layer
        representation:

            [X -> [p_in] -> relu([h_dim]) -> [p_out]]

        Args:
            p_in  (int): the input dimensionality
            p_out (int): the desired output dimensionality (compression
                         dimensionality)
        """

        super(HalfAutoEncoder, self).__init__()
        self.l1 = nn.Linear(p_in, h_dim)
        self.l2 = nn.Linear(h_dim, p_out)
        self.activation = torch.relu

    def forward(self, x):
        h1 = self.activation(self.l1(x))
        return self.l2(h1)

    def get_parameters(self):
        return [param for param in self.named_parameters()]


class EmbeddingGP(nn.Module):
    """An Embedding-to-GP model that trains an input embedding and output GP
    at the same time.

    Attributes:
        net (nn.Module): the compressing neural network.
        p_compression (int): the dimensionality of the neural net output
                             (the compression dimensionality)
        GP (pyro.contrib.gp.models.vgp.VariationalGP): the GP
    """

    def __init__(self, net, X, y):
        """
        An Embedding-to-GP model. This model takes an arbitrary neural network
        as an input and adds a GP at the last layer. This trains the GP
        hyperparameters and the neural net parameters simultaneously through
        backprop, learning an optimal compression for predictive purposes.

        Args:
            net (nn.Module): A PyTorch neural network for compressing
                                  input data.
            X (torch.Tensor): input data
            y (torch.Tensor): output target data
        """

        super(EmbeddingGP, self).__init__()
        self.net = net
        self.p_compression = net.get_parameters()[-1][1].shape[0]
        self.GP = self._get_GP(X, y)

    def _get_GP(self, X, y):
        """
        Defines a Variational Gaussian Process with a Warping kernel that takes
        a neural net as an input.

        Parameter notes:
            - PSD criterion is sensitive to jitter-level
            - Unclear of the right base kernel hyperparameter initializations
        """

        kernel_base = gp.kernels.RBF(self.p_compression,
                                     variance=torch.tensor(1.),
                                     lengthscale=torch.tensor(1.))
        kernel_deep = gp.kernels.Warping(kernel_base, iwarping_fn=self.net)
        likelihood = gp.likelihoods.Gaussian(variance=torch.tensor(1.))
        return gp.models.VariationalGP(X=X,
                                       y=y,
                                       kernel=kernel_deep,
                                       likelihood=likelihood,
                                       whiten=True,
                                       jitter=1e-4)

    def train(self, n_steps=10000, print_steps=100, plot=True):
        """
        Train the embedding and GP.

        Args:
            n_steps (int): number of training steps.
            print_steps (int): print the loss if step is a multiple of
                              `print_steps`. `None` if no print.
            plot (bool): plot the steps-by-loss matrix after training.
        """

        pyro.clear_param_store()
        learning_rate = 0.2 * 1e-2
        momentum = 1e-1
        optimizer = torch.optim.SGD(self.GP.parameters(),
                                    lr=learning_rate,
                                    momentum=momentum)
        optimizer = torch.optim.Adam(self.GP.parameters(), lr=learning_rate)
        elbo = pyro.infer.TraceMeanField_ELBO()
        loss_fn = elbo.differentiable_loss
        n_steps = n_steps

        # optimize
        losses = []
        for i in range(1, n_steps + 1):
            optimizer.zero_grad()
            loss = loss_fn(self.GP.model, self.GP.guide)
            if print_steps is not None and i % print_steps == 0:
                print("Step {}: {}".format(i, loss))
            losses.append(loss)
            loss.backward()
            optimizer.step()

        self.losses = losses
        if plot:
            self._plot()

    def _plot(self):
        plt.plot(torch.arange(len(self.losses)).numpy(), self.losses)
        plt.xlabel("Steps")
        plt.ylabel("Loss")

    def predict(self, X):
        return self.GP(X)


def main():
    n = 1000
    p = 3
    X, y = generate_data(n, p)
    net = HalfAutoEncoder(p, 2, 2)
    egp = EmbeddingGP(net, X, y.squeeze(-1))
    egp.train(print_steps=10)
    return egp


if __name__ == "__main__":
    main()
