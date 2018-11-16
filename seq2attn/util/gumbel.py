"""Gumbel-Softmax (with and without Straight-Through Estimator.

based on
https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
"""

import torch
from torch.nn.functional import softmax


def _sample_gumbel(shape, eps, out=None):
    """Sample from Gumbel(0, 1)."""
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return - torch.log(eps - torch.log(U + eps))


def _gumbel_softmax_sample(logits, invalid_action_mask, gumbel, tau, eps):
    """Draw a sample from the Gumbel-Softmax distribution."""
    dims = logits.dim()
    gumbel_noise = _sample_gumbel(logits.size(), eps=eps, out=logits.data.new())

    if gumbel:
        y = logits + gumbel_noise
    else:
        y = logits

    # Make temperature same dimensions as y and
    # make certain values non-differentiable by filling their content.
    # Otherwise, pytorch would try to differentiate,
    # y would contain -inf values, making y/tau produce NaN gradients
    # We need to use contiguous to make sure that masked_fill_ does not fill all copied values,
    # but only the ones indicated by the mask.
    # We expect here that y has already been masked with -inf with the same mask
    tau = tau.expand_as(y).contiguous()
    tau.masked_fill_(mask=invalid_action_mask, value=1)

    return softmax(y / tau, dims - 1)


def gumbel_softmax(logits, invalid_action_mask, tau, gumbel, hard, eps):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.

    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now
    """
    shape = logits.size()
    assert len(shape) == 2
    y_soft = _gumbel_softmax_sample(logits, invalid_action_mask, gumbel=gumbel, tau=tau, eps=eps)
    if hard:
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = logits.data.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = (y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y, y_soft
