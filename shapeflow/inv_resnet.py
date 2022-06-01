import torch
import normflow as nf
import torch.nn


def get_residual_transform(
    shape: torch.Size,
    hidden_features: int,
    hidden_layers: int = None,
    kernel_size: int = None,
    CNN: bool = False,
    n_exact_terms=2,
    n_samples=1,
    reduce_memory=True,
    reverse=True,
):
    latent_size = shape[0]
    if CNN:

        if kernel_size is None:
            kernel_size = hidden_features
        assert kernel_size % 2 == 1, f"kernel size must be odd but is {kernel_size}"
        net = nf.nets.LipschitzCNN(
            channels=[1] * (hidden_layers + 1),
            kernel_size=[kernel_size] * (hidden_layers),
            init_zeros=True,
            lipschitz_const=0.9,
        )
    else:
        net = nf.nets.LipschitzMLP(
            [latent_size] + [hidden_features] * (hidden_layers - 1) + [latent_size],
            init_zeros=True,
            lipschitz_const=0.9,
        )
    if conditional:
        pre_net = InducedNormLinear(
                    in_features=hidden_layers,
                    out_features=channels[i + 1],
                    coeff=lipschitz_const,
                    domain=2,
                    codomain=2,
                    n_iterations=max_lipschitz_iter,
                    atol=lipschitz_tolerance,
                    rtol=lipschitz_tolerance,
                    zero_init=init_zeros if i == (self.n_layers - 1) else False,
                ),
        transform = nf.flows.Residual(
            net,
            n_exact_terms=n_exact_terms,
            n_samples=n_samples,
            reduce_memory=reduce_memory,
            reverse=reverse,
        )
    else
        transform = nf.flows.Residual(
            net,
            n_exact_terms=n_exact_terms,
            n_samples=n_samples,
            reduce_memory=reduce_memory,
            reverse=reverse,
        )
    return transform



# Try importing Residual Flow dependencies
try:
    from residual_flows.layers import iResBlock
except:
    print('Warning: Dependencies for Residual Flows could '
          'not be loaded. Other models can still be used.')



# class ConditionalResidual(torch.nn.Module):
#     """
#     Invertible residual net block, wrapper to the implementation of Chen et al. not with context,
#     see https://github.com/rtqichen/residual-flows
#     """
#     def __init__(self, inv_net, pre_inv_net, net: torch.nn.Module,n_exact_terms=2, n_samples=1, reduce_memory=True,
#                  reverse=True):
#         """
#         Constructor
#         :param net: Neural network, must be Lipschitz continuous with L < 1
#         :param n_exact_terms: Number of terms always included in the power series
#         :param n_samples: Number of samples used to estimate power series
#         :param reduce_memory: Flag, if true Neumann series and precomputations
#         for backward pass in forward pass are done
#         :param reverse: Flag, if true the map f(x) = x + net(x) is applied in
#         the inverse pass, otherwise it is done in forward
#         """
#         super().__init__()
#         self.reverse = reverse
#         self.pre_iresblock = iResBlock(pre_inv_net, n_samples=n_samples,
#                                    n_exact_terms=n_exact_terms,
#                                    neumann_grad=reduce_memory,
#                                    grad_in_forward=reduce_memory) if pre_inv_net is not None else None
#
#         self.iresblock = iResBlock(inv_net, n_samples=n_samples,
#                                    n_exact_terms=n_exact_terms,
#                                    neumann_grad=reduce_memory,
#                                    grad_in_forward=reduce_memory)
#         self.net = net
#
#     def forward(self, z:torch.Tensor, context:torch.Tensor = None):
#         # account for distortion by first block
#         if self.pre_iresblock is not None:
#             if self.reverse:
#                 z, log_det = self.iresblock.inverse(z, 0)
#             else:
#                 z, log_det = self.iresblock.forward(z, 0)
#             log_det_pre = log_det.view(-1)
#         else:
#             log_det_pre = 0
#
#         if context is not None:
#             z_mod = self.net(context)
#             z += + z_mod
#
#         if self.reverse:
#             z, log_det = self.iresblock.inverse(z, 0)
#         else:
#             z, log_det = self.iresblock.forward(z, 0)
#
#         return z, -log_det.view(-1) - log_det_pre
#
#     def inverse(self, z, context):
#         # do everything in reverse
#         if self.reverse:
#             z, log_det = self.iresblock.forward(z, 0)
#         else:
#             z, log_det = self.iresblock.inverse(z, 0)
#
#         if context is not None:
#             z_mod = self.net(context)
#             z -= z_mod
#
#         if self.pre_iresblock is not None:
#             if self.reverse:
#                 z, log_det_pre = self.iresblock.forward(z, 0)
#             else:
#                 z, log_det_pre = self.iresblock.inverse(z, 0)
#             log_det_pre= log_det_pre.view(-1)
#         else:
#             log_det_pre = 0
#
#         return z, -log_det.view(-1) - log_det_pre
