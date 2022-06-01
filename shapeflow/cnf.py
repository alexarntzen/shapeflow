def autograd_trace(x_out, x_in, **kwargs):
    """Standard brute-force means of obtaining trace of the Jacobian, O(d) calls to autograd"""
    trJ = 0.
    for i in range(x_in.shape[1]):
        trJ += torch.autograd.grad(x_out[:, i].sum(), x_in, allow_unused=False, create_graph=True)[0][:, i]
    return trJ


class CNF(nn.Module):
    def __init__(self, net, trace_estimator=None, noise_dist=None):
        super().__init__()
        self.net = net
        self.trace_estimator = trace_estimator if trace_estimator is not None else autograd_trace;
        self.noise_dist, self.noise = noise_dist, None

    def forward(self, x):
        with torch.set_grad_enabled(True):
            x_in = x[:, 1:].requires_grad_(True)  # first dimension reserved to divergence propagation
            # the neural network will handle the data-dynamics here
            x_out = self.net(x_in)

            trJ = self.trace_estimator(x_out, x_in, noise=self.noise)
        return torch.cat([-trJ[:, None], x_out],
                         1) + 0 * x  # `+ 0*x` has the only purpose of connecting x[:, 0] to autograd graph