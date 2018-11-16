"""Seq2Attn Decoder.

Implements both the transcoder and decoder.
The decoder is initialized with a learnable vector and receives input from the transcoder
in the form of attention over the encoder states or input embeddings.
"""

import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .attention import HardGuidance
from .attentionActivation import AttentionActivation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Seq2AttnDecoder(nn.Module):
    """seq2attn model with attention.

    First, pass the input sequence to `select_actions()` to perform forward pass and retrieve the
    actions.
    Next, calculate and pass the rewards for the selected actions.
    Finally, call `finish_episod()` to calculate the discounted rewards and policy loss.
    """

    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.

    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)

    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`

    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).

    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, vocab_size, max_len, hidden_size,
                 sos_id, eos_id, embedding_dim,
                 n_layers=1, rnn_cell='gru', bidirectional=False,
                 input_dropout_p=0, dropout_p=0, use_attention=False, attention_method=None,
                 sample_train=None,
                 sample_infer=None,
                 initial_temperature=None,
                 learn_temperature=None,
                 attn_vals=None,
                 full_attention_focus='no'):
        super(Seq2AttnDecoder, self).__init__()

        # Store values
        self.bidirectional_encoder = bidirectional
        self.rnn_type = rnn_cell
        self.max_length = max_len
        self.hidden_size = hidden_size
        self.use_attention = use_attention
        self.n_layers = n_layers
        self.eos_id = eos_id
        self.sos_id = sos_id
        self.full_attention_focus = (full_attention_focus == 'yes')

        # Get type of RNN cell
        rnn_cell = rnn_cell.lower()
        self.rnn_type = rnn_cell
        if rnn_cell == 'lstm':
            rnn_cell = nn.LSTM
        elif rnn_cell == 'gru':
            rnn_cell = nn.GRU

        # Store pointer to attention keys and values
        self.attn_keys = 'encoder_outputs'
        if attn_vals == 'embeddings':
            self.attn_vals = 'encoder_embeddings'
        elif attn_vals == 'outputs':
            self.attn_vals = 'encoder_outputs'

        # Store attention method
        self.use_attention = use_attention
        if self.use_attention != 'pre-rnn':
            raise Exception("Must use pre-rnn in combination with seq2attn")

        # Create learnable parameter for initializing the decoder
        if self.rnn_type == 'lstm':
            self.decoder_hidden0 = (
                nn.Parameter(torch.zeros([self.n_layers, 1, self.hidden_size], device=device)),
                nn.Parameter(torch.zeros([self.n_layers, 1, self.hidden_size], device=device)))
        elif self.rnn_type == 'gru':
            self.decoder_hidden0 = nn.Parameter(torch.zeros([self.n_layers, 1, self.hidden_size],
                                                            device=device))

        # Input size for the decoder is hidden_size + context vector size,
        # which depends on the type of attention value.
        # Input size for MLP attention is concatenation of hidden size and attention key size.
        key_dim = hidden_size
        if 'embeddings' in attn_vals:
            val_dim = embedding_dim
        elif 'outputs' in attn_vals:
            val_dim = hidden_size
        decoder_input_size = key_dim + val_dim
        transcoder_input_size = hidden_size
        attention_input_size = hidden_size + key_dim

        # Initialize model
        self.embedding = nn.Embedding(
            vocab_size,
            hidden_size)
        self.input_dropout = nn.Dropout(
            p=input_dropout_p)
        self.transcoder = rnn_cell(
            transcoder_input_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout_p)
        attention_activation = AttentionActivation(
            sample_train=sample_train,
            sample_infer=sample_infer,
            learn_temperature=learn_temperature,
            initial_temperature=initial_temperature,
            query_dim=hidden_size)
        self.attention = Attention(
            input_dim=attention_input_size,
            hidden_dim=hidden_size,
            method=attention_method,
            attention_activation=attention_activation)
        self.decoder = rnn_cell(
            decoder_input_size,
            hidden_size,
            n_layers,
            batch_first=True,
            dropout=dropout_p)
        self.out = nn.Linear(self.hidden_size, vocab_size)

    def get_valid_action_mask(self, state, input_lengths):
        """Get valid action mask.

        Get a bytetensor that indicates which encoder states are valid to attend to.
        All <pad> steps are invalid

        Args:
            state (torch.tensor): [batch_size x max_input_length] input variable
            input_lengths (torch.tensor): [batch_size] tensor containing the input length of each
                                          sequence in the batch

        Returns:
            torch.tensor: [batch_size x max_input_length] ByteTensor with a 0 for
                          all <pad> elements

        """
        batch_size = state.size(0)

        # First, we establish which encoder states are valid to attend to. For
        # this we use the input_lengths
        max_encoding_length = torch.max(input_lengths)

        # (batch_size) -> (batch_size x max_encoding_length)
        input_lengths_expanded = input_lengths.unsqueeze(1).expand(-1, max_encoding_length)

        # Use arange to create list 0, 1, 2, 3, .. for each element in the batch
        # (batch_size x max_encoding_length)
        encoding_steps_indices = torch.arange(max_encoding_length, dtype=torch.long, device=device)
        encoding_steps_indices = encoding_steps_indices.unsqueeze(0).expand(batch_size, -1)

        # A (batch_size x max_encoding_length) tensor that has a 1 for all valid
        # actions and 0 for all invalid actions
        valid_action_mask = encoding_steps_indices < input_lengths_expanded

        return valid_action_mask

    def forward_decoder(self, embedded, transcoder_hidden, decoder_hidden, attn_keys, attn_vals,
                        **attention_method_kwargs):
        """Forward decoder.

        Perform forward pass and stochastically select actions using epsilon-greedy RL

        Args:
            state (torch.tensor): [batch_size x max_input_length] tensor containing indices of the
                                  input sequence
            input_lengths (list): List containing the input length for each element in the batch
            max_decoding_length (int): Maximum length till which the decoder should run
            epsilon (float): epsilon for epsilon-greedy RL. Set to 1 in inference mode

        Returns:
            list(torch.tensor): List of length max_output_length containing the selected actions

        """
        transcoder_output, transcoder_hidden = self.transcoder(embedded, transcoder_hidden)
        context, attn = self.attention(queries=transcoder_output, keys=attn_keys, values=attn_vals,
                                       **attention_method_kwargs)
        decoder_input = torch.cat((context, embedded), dim=2)

        if self.full_attention_focus:
            if self.rnn_type == 'gru':
                decoder_hidden = decoder_hidden * context.transpose(0, 1)
            elif self.rnn_type == 'lstm':
                decoder_hidden = (decoder_hidden[0] * context.transpose(0, 1),
                                  decoder_hidden[1] * context.transpose(0, 1))
        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

        output = decoder_output

        return output, transcoder_hidden, decoder_hidden, attn

    def forward_step(self, input_var, transcoder_hidden, decoder_hidden, attn_keys, attn_vals,
                     function, **attention_method_kwargs):
        """One forward step.

        Performs one or multiple forward decoder steps.

        Args:
            input_var (torch.tensor): Variable containing the input(s) to the decoder RNN
            hidden (torch.tensor): Variable containing the previous decoder hidden state.
            encoder_outputs (torch.tensor): Variable containing the target outputs of the
                                            decoder RNN

        Returns:
            predicted_softmax: The output softmax distribution at every time step of the
                               decoder RNN
            hidden: The hidden state at every time step of the decoder RNN
            attn: The attention distribution at every time step of the decoder RNN

        """
        batch_size = input_var.size(0)
        output_size = input_var.size(1)

        embedded = self.embedding(input_var)
        embedded = self.input_dropout(embedded)

        return_values = self.forward_decoder(
            embedded,
            transcoder_hidden,
            decoder_hidden,
            attn_keys=attn_keys,
            attn_vals=attn_vals,
            **attention_method_kwargs)

        output = return_values[0].contiguous().view(batch_size, -1)
        output = self.out(output)
        activated_output = function(output, dim=1).view(batch_size, output_size, -1)

        new_return_values = [activated_output]
        for i in range(1, len(return_values)):
            new_return_values.append(return_values[i])

        return new_return_values

    def forward(self, inputs=None,
                encoder_embeddings=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0, provided_attention=None):
        """Forward."""
        ret_dict = dict()
        if self.use_attention:
            ret_dict[Seq2AttnDecoder.KEY_ATTN_SCORE] = list()

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden,
                                                             encoder_outputs,
                                                             teacher_forcing_ratio)

        transcoder_hidden = self._init_state(encoder_hidden, 'encoder')
        decoder_hidden = self._init_state(encoder_hidden, 'new')

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[Seq2AttnDecoder.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        # Prepare extra arguments for attention method
        attention_method_kwargs = {}
        if self.attention and isinstance(self.attention.method, HardGuidance):
            attention_method_kwargs['provided_attention'] = provided_attention

        # When we use pre-rnn attention we must unroll the decoder. We need to calculate the
        # attention based on the previous hidden state, before we can calculate the next
        # hidden state.
        # We also need to unroll when we don't use teacher forcing. We need perform the
        # decoder steps one-by-one since the output needs to be copied to the input of
        # the next step.
        if self.use_attention == 'pre-rnn' or not use_teacher_forcing:
            unrolling = True
        else:
            unrolling = False

        # Get local variable out of locals() dictionary by string key
        attn_keys = locals()[self.attn_keys]
        attn_vals = locals()[self.attn_vals]

        if unrolling:
            symbols = None
            for di in range(max_length):
                # We always start with the SOS symbol as input. We need to add extra dimension of
                # length 1 for the number of decoder steps (1 in this case).
                # When we use teacher forcing, we always use the target input.
                if di == 0 or use_teacher_forcing:
                    decoder_input = inputs[:, di].unsqueeze(1)
                # If we don't use teacher forcing (and we are beyond the first SOS step), we use
                # the last output as new input
                else:
                    decoder_input = symbols

                # Perform one forward step
                if self.attention and isinstance(self.attention.method, HardGuidance):
                    attention_method_kwargs['step'] = di

                decoder_output, transcoder_hidden, decoder_hidden, step_attn = self.forward_step(
                    decoder_input,
                    transcoder_hidden,
                    decoder_hidden,
                    attn_keys,
                    attn_vals,
                    function,
                    **attention_method_kwargs)

                # Remove the unnecessary dimension.
                step_output = decoder_output.squeeze(1)
                # Get the actual symbol
                symbols = decode(di, step_output, step_attn)

        else:
            # Remove last token of the longest output target in the batch. We don't have to run
            # the last decoder step where the teacher forcing input is EOS (or the last output)
            # It still is run for shorter output targets in the batch
            decoder_input = inputs[:, :-1]

            # Forward step without unrolling
            if self.attention and isinstance(self.attention.method, HardGuidance):
                attention_method_kwargs['step'] = -1

            decoder_output, transcoder_hidden, decoder_hidden, attn = self.forward_step(
                decoder_input,
                transcoder_hidden,
                decoder_hidden,
                attn_keys,
                attn_vals,
                function,
                **attention_method_kwargs)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)

        ret_dict[Seq2AttnDecoder.KEY_SEQUENCE] = sequence_symbols
        ret_dict[Seq2AttnDecoder.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden, init_dec_with):
        if init_dec_with == 'encoder':
            """ Initialize the encoder hidden state. """
            if encoder_hidden is None:
                return None
            if isinstance(encoder_hidden, tuple):
                encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
            else:
                encoder_hidden = self._cat_directions(encoder_hidden)

        elif init_dec_with == 'new':
            if isinstance(self.decoder_hidden0, tuple):
                batch_size = encoder_hidden[0].size(1)
                encoder_hidden = (
                    self.decoder_hidden0[0].repeat(1, batch_size, 1),
                    self.decoder_hidden0[1].repeat(1, batch_size, 1))
            else:
                batch_size = encoder_hidden.size(1)
                encoder_hidden = self.decoder_hidden0.repeat(1, batch_size, 1)

        return encoder_hidden

    def _cat_directions(self, h):
        """If the encoder is bidirectional, do the following transformation.

        (#directions * #layers, #batch, hidden_size) ->
            (#layers, #batch, #directions * hidden_size)

        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_type == 'lstm':
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_type == 'gru':
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError(
                    "Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.tensor([self.sos_id] * batch_size, dtype=torch.long, device=device)
            inputs = inputs.view(batch_size, 1)

            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1  # minus the start of sequence symbol

        return inputs, batch_size, max_length
