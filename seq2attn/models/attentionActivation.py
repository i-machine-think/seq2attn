"""Activation/sampling functionality for attention vector."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..util.gumbel import gumbel_softmax
from ..util.sparsemax import Sparsemax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AttentionActivation(nn.Module):
    """Activation/sampling functionality for attention vector."""

    def __init__(self, sample_train='softmax', sample_infer='softmax',
                 learn_temperature='no', initial_temperature=0., query_dim=0):
        """Initialize activation function."""
        super(AttentionActivation, self).__init__()

        self.sample_train = sample_train
        self.sample_infer = sample_infer
        self.query_dim = query_dim

        # Initialize temperature
        self.current_temperature = None

        if 'gumbel' in sample_train or sample_train == 'softmax_st':
            self.learn_temperature = learn_temperature

            if learn_temperature == 'no':
                self.temperature = torch.tensor(initial_temperature,
                                                requires_grad=False, device=device)

            elif learn_temperature == 'latent':
                self.temperature = nn.Parameter(torch.log(torch.tensor(initial_temperature,
                                                                       device=device)),
                                                requires_grad=True)
                self.temperature_activation = torch.exp

            elif learn_temperature == 'conditioned':
                self.max_temperature = initial_temperature

                self.inverse_temperature_estimator = nn.Linear(query_dim, 1)
                self.inverse_temperature_activation = self.inverse_temperature_activation

        # Initialize sparsemax
        if self.sample_train == 'sparsemax' or self.sample_infer == 'sparsemax':
            self.sparsemax = Sparsemax()

    def inverse_temperature_activation(self, inv_temp):
        """Calculate the inverse temperature for the 'conditioned' condition."""
        inverse_max_temperature = 1. / self.max_temperature
        return torch.log(1 + torch.exp(inv_temp)) + inverse_max_temperature

    def update_temperature(self, queries):
        """Update current temperature."""
        if self.learn_temperature == 'no':
            self.current_temperature = self.temperature

        elif self.learn_temperature == 'latent':
            self.current_temperature = self.temperature_activation(self.temperature)

        elif self.learn_temperature == 'conditioned':
            batch_size, max_decoder_length, hidden_dim = queries.size()
            assert hidden_dim == self.query_dim, "Query dim is not set correctly."

            estimator_input = queries.view(batch_size * max_decoder_length, hidden_dim)
            inverse_temperature = self.inverse_temperature_activation(
                                    self.inverse_temperature_estimator(estimator_input))
            self.current_temperature = 1. / inverse_temperature

    def sample(self, attn, mask):
        """Sample/activate attention vector."""
        batch_size, output_size, input_size = attn.size()

        # We are in training mode
        if self.training:
            if self.sample_train == 'softmax':
                attn = F.softmax(attn, dim=2)

            elif self.sample_train == 'softmax_st':
                attn = F.log_softmax(attn.view(-1, input_size), dim=1)

                mask = mask.expand(batch_size, output_size, input_size).contiguous()
                mask = mask.view(-1, input_size)
                attn_hard, attn_soft = gumbel_softmax(
                    logits=attn, invalid_action_mask=mask, hard=True, tau=self.current_temperature,
                    gumbel=False, eps=1e-20)
                attn = attn_hard.view(batch_size, -1, input_size)

            elif 'gumbel' in self.sample_train:
                attn = F.log_softmax(attn.view(-1, input_size), dim=1)
                mask = mask.expand(batch_size, output_size, input_size).contiguous()
                mask = mask.view(-1, input_size)
                attn_hard, attn_soft = gumbel_softmax(
                    logits=attn, invalid_action_mask=mask, hard=True, tau=self.current_temperature,
                    gumbel=True, eps=1e-20)

                if self.sample_train == 'gumbel':
                    attn = attn_soft.view(batch_size, -1, input_size)
                elif self.sample_train == 'gumbel_st':
                    attn = attn_hard.view(batch_size, -1, input_size)

            elif self.sample_train == 'sparsemax':
                # Sparsemax only handles 2-dim tensors,
                # so we reshape and reshape back after sparsemax
                original_size = attn.size()
                attn = attn.view(-1, attn.size(2))
                attn = self.sparsemax(attn)
                attn = attn.view(original_size)

        # Inference mode
        else:
            if self.sample_infer == 'softmax':
                attn = F.softmax(attn, dim=2)

            elif self.sample_infer == 'softmax_st':
                attn = F.log_softmax(attn.view(-1, input_size), dim=1)
                mask = mask.expand(batch_size, output_size, input_size).contiguous()
                mask = mask.view(-1, input_size)
                attn_hard, attn_soft = gumbel_softmax(
                    logits=attn, invalid_action_mask=mask,
                    hard=True, tau=self.current_temperature, gumbel=False, eps=1e-20)
                attn = attn_hard.view(batch_size, -1, input_size)

            elif 'gumbel' in self.sample_infer:
                attn = F.log_softmax(attn.view(-1, input_size), dim=1)
                mask = mask.expand(batch_size, output_size, input_size).contiguous()
                mask = mask.view(-1, input_size)
                attn_hard, attn_soft = gumbel_softmax(
                    logits=attn, invalid_action_mask=mask, hard=True, tau=self.current_temperature,
                    gumbel=True, eps=1e-20)

                if self.sample_infer == 'gumbel':
                    attn = attn_soft.view(batch_size, -1, input_size)
                elif self.sample_infer == 'gumbel_st':
                    attn = attn_hard.view(batch_size, -1, input_size)

            elif self.sample_infer == 'argmax':
                argmax = attn.argmax(dim=2, keepdim=True)
                attn = torch.zeros_like(attn)
                attn.scatter_(dim=2, index=argmax, value=1)

            elif self.sample_infer == 'sparsemax':
                # Sparsemax only handles 2-dim tensors,
                # so we reshape and reshape back after sparsemax
                original_size = attn.size()
                attn = attn.view(-1, attn.size(2))
                attn = self.sparsemax(attn)
                attn = attn.view(original_size)

        return attn

    def forward(self, attn, mask, queries):
        """Forward function."""
        # Update temperature
        if 'gumbel' in self.sample_train or self.sample_train == 'softmax_st' or \
           'gumbel' in self.sample_infer or self.sample_infer == 'softmax_st':
                self.update_temperature(queries)

        # Sample attention vector
        attn = self.sample(attn, mask)

        # Check whether attention vectors sum up to 1
        number_of_attention_vectors = attn.size(0) * attn.size(1)
        eps = 1e-2 * number_of_attention_vectors
        assert abs(torch.sum(attn) - number_of_attention_vectors) < eps, \
            "Sum: {}, Number of attention vectors: {}".format(torch.sum(attn),
                                                              number_of_attention_vectors)

        return attn
