import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .blocks import TransformerBlock
from .position import positional_encoding_vector


class BaseModule(nn.Module):
    """Wrapper around nn.Module that recursively sets history and embedding usage.

    All modules should inherit from this class.
    """
    def __init__(self):
        super().__init__()
        self._using_history = False
        self._using_embedding = False
        self._using_marks = False

    @property
    def using_history(self):
        return self._using_history

    @property
    def using_embedding(self):
        return self._using_embedding

    @property
    def using_marks(self):
        return self._using_marks

    def use_history(self, mode=True):
        """Recursively make all submodules use history."""
        self._using_history = mode
        for module in self.children():
            if isinstance(module, BaseModule):
                module.use_history(mode)

    def use_embedding(self, mode=True):
        """Recursively make all submodules use embeddings."""
        self._using_embedding = mode
        for module in self.children():
            if isinstance(module, BaseModule):
                module.use_embedding(mode)

    def use_marks(self, mode=True):
        """Recursively make all submodules use embeddings."""
        self._using_marks = mode
        for module in self.children():
            if isinstance(module, BaseModule):
                module.use_marks(mode)



class RNNLayer(BaseModule):
    """RNN for encoding the event history."""
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.history_size
        self.rnn_type = config.rnn_type
        self.use_history(config.use_history)
        self.use_marks(config.use_marks)
        self.device = config.device

        if config.use_marks:
            # Define mark embedding layer
            self.mark_embedding = nn.Embedding(config.num_classes, config.mark_embedding_size)
            # If we have marks, input is time + mark embedding vector
            self.in_features = config.mark_embedding_size + 1
        else:
            # Without marks, input is only time
            self.in_features = 1

        # Possible RNN types: 'RNN', 'GRU', 'LSTM'
        self.rnn = getattr(nn, self.rnn_type)(self.in_features, self.hidden_size, batch_first=True)

    def forward(self, input):
        """Encode the history of the given batch.

        Returns:
            h: History encoding, shape (batch_size, seq_len, self.hidden_size)
        """
        t = input.in_time
        length = input.length

        if not self.using_history:
            return torch.zeros(t.shape[0], t.shape[1], self.hidden_size)

        x = t.unsqueeze(-1)
        if self.using_marks:
            mark = self.mark_embedding(input.in_mark)
            x = torch.cat([x, mark], -1)

        h_shape = (1, x.shape[0], self.hidden_size)
        if self.rnn_type == 'LSTM':
            # LSTM keeps two hidden states
            h0 = (torch.zeros(h_shape), torch.zeros(h_shape))
        else:
            # RNN and GRU have one hidden state
            h0 = torch.zeros(h_shape)
        h0 = h0.to(self.device)

        x, batch_sizes = torch._C._VariableFunctions._pack_padded_sequence(x, length.cpu().long(), batch_first=True)
        x = torch.nn.utils.rnn.PackedSequence(x, batch_sizes)

        h, _ = self.rnn(x, h0)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        return h

    def step(self, x, h):
        """Given input and hidden state produces the output and new state."""
        y, h = self.rnn(x, h)
        return y, h
    

class AMDN_RNNLayer(BaseModule):
    """RNN for encoding the event history."""
    def __init__(self, history_size, rnn_type, device):
        super().__init__()
        self.device = device
        self.hidden_size = history_size
        self.rnn_type = rnn_type
        # Possible RNN types: 'RNN', 'GRU', 'LSTM'
        self.rnn = getattr(nn, self.rnn_type)(self.hidden_size, self.hidden_size, batch_first=True)

        
    def forward(self, x, length):
        """Encode the history of the given batch.

        Returns:
            h: History encoding, shape (batch_size, seq_len, self.hidden_size)
        """
        # x = attention output (b,t,e)  # length = input.length
        h_shape = (1, x.shape[0], self.hidden_size)
        
        if self.rnn_type == 'LSTM':
            h0 = (torch.zeros(h_shape), torch.zeros(h_shape))  # LSTM keeps two hidden states
        else:
            h0 = torch.zeros(h_shape)  # RNN and GRU have one hidden state
        h0 = h0.to(self.device)
            
        x, batch_sizes = torch._C._VariableFunctions._pack_padded_sequence(x, length.cpu().long(), batch_first=True)
        x = torch.nn.utils.rnn.PackedSequence(x, batch_sizes)

        h, _ = self.rnn(x, h0)
        h, _ = torch.nn.utils.rnn.pad_packed_sequence(h, batch_first=True)
        return h

    
    def step(self, x, h):
        """Given input and hidden state produces the output and new state."""
        y, h = self.rnn(x, h)
        return y, h
    
    
class AttentiveLayer(BaseModule):
    """Self-attention for encoding the event history."""
    
    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.add = config.add
        self.time_opt = config.time_opt
        self.use_marks(config.use_marks)
        self.use_community = config.use_community
        self.in_features = 1  # Without marks, input is only time  - max seq_length
        if config.use_marks: 
            self.mark_embedding = nn.Embedding(config.num_classes, config.mark_embedding_size)
            if config.use_community:
                self.community_embedding = nn.Embedding(config.gmm_k, config.mark_embedding_size)
                self.gmm_k = config.gmm_k
            print(config.num_classes, config.mark_embedding_size)
            if config.pos_enc: 
                self.in_features = config.mark_embedding_size + 1
            else:
                self.in_features = config.mark_embedding_size
        
        # self.pos_embedding = nn.Embedding(config.seq_length, self.in_features)
        self.pos_enc_config = config.pos_enc
        self.pos_encoding = torch.FloatTensor(positional_encoding_vector(
                self.in_features, config.seq_length, dtype=np.float32)).to(self.device)  #(seq_length, in_features size)
        
        tblocks = []
        for i in range(config.depth):
            if i == config.depth - 1: last_block = True
            else: last_block = False
            tblocks.append(
                TransformerBlock(
                    emb=config.history_size, heads=config.heads, seq_length=config.seq_length, mask=True,
                    wide=config.wide, last_block=last_block)
            )
        self.tblocks = nn.Sequential(*tblocks)
        
        # time interval encoding (no need for positional encoding, input time as feature can be removed)
        # t_bs, t_bs_freq = basis_time_encode(self.input_t, return_weight=True)
        self.time_dim = config.mark_embedding_size  # int(embed_units * time_factor)
        self.expand_dim = config.expand_dim  # k for expand_factor
        init_period_base = np.array(np.linspace(0, 8, self.time_dim), dtype=np.float32)
        self.period_var = 10.0 ** torch.FloatTensor(init_period_base)#.to(self.device)
        self.period_var = self.period_var[:, None].expand(
            self.time_dim, self.expand_dim)  # [time_dim] -> [time_dim, 1] -> [time_dim, expand_dim]
        self.expand_coef = torch.FloatTensor(
            np.arange(self.expand_dim) + 1).reshape(1, -1)#.to(self.device)  
        # tf.reshape(tf.range(expand_dim) + 1, [1, -1])

        self.freq_var = 1 / self.period_var
        self.freq_var = self.freq_var * self.expand_coef
        self.freq_var = self.freq_var.unsqueeze(0).unsqueeze(0)

        self.basis_expan_var = torch.nn.init.xavier_uniform_(
            torch.empty((self.time_dim, 2*self.expand_dim))).unsqueeze(0).unsqueeze(0).to(self.device)  
        self.basis_expan_var_bias = torch.zeros((self.time_dim)).unsqueeze(0).unsqueeze(0).to(self.device) 
        # glorot uniform init

        # RNN over attention outputs for next event prediction (Avoid using this option)
        self.encoder_type = config.encoder_type
        if self.encoder_type == 'ATTN_RNN':
            self.rnn = AMDN_RNNLayer(config.history_size, config.rnn_type, config.device)
        
    def forward(self, input, gmm_prob = None):
        """
        Encode the history of the given batch. (TIME as input feature, position in seq emb and concat with [mark,time])
        Returns:
            h:  History encoding, shape (batch_size, seq_len, self.hidden_size)
                hidden states at each time input in input sequences.
        """
        if self.pos_enc_config:  # positional encoding 
            x = self.positional_encoding(input)
        else:  # time interval encoding
            #if gmm_prob is not None:
            mark_tokens = self.mark_embedding(input.in_mark)
            b, t, e = mark_tokens.size()
            #community_tokens = None
            if gmm_prob is not None and self.use_community:
                if len(gmm_prob.size())== 2:
                    community_tokens = self.community_embedding(gmm_prob.long())
                else:
                    community_tokens = torch.sum(gmm_prob.unsqueeze(-1)*self.community_embedding.weight,dim = -2).float()
                    #print(community_tokens.size())
            if self.time_opt == 'cumsum_exp':
                self.dt_attn_inputs = torch.cumsum(torch.exp(input.in_time), dim=-1)  # (b, t)
            elif self.time_opt == 'cumsum':
                self.dt_attn_inputs = torch.cumsum(input.in_time, dim=-1)
            elif self.time_opt == 'delta':
                self.dt_attn_inputs = input.in_time
            else:
                print('Not implemented time_opt')
            # the padding part will be ignored later in loss compute function etc should be
            time_embedding = self.basis_time_encode(self.dt_attn_inputs)
            b, t, e = mark_tokens.size()
            positions = self.pos_encoding[None, :t, :].expand(b, t, e)
            if not self.add:
                if self.use_community:
                    if gmm_prob is not None:
                        '''print('time',time_embedding.size())
                        print('pos',positions.size())
                        print('community',community_tokens.size())
                        print('marks',mark_tokens.size())'''
                        x = torch.cat([mark_tokens, community_tokens, time_embedding, positions], dim=2)
                    else:
                        x = torch.cat([mark_tokens, torch.zeros_like(mark_tokens).to(self.device), time_embedding, positions], dim=2)
                else:
                    x = torch.cat([mark_tokens, time_embedding, positions], dim=2)
            else:
                if gmm_prob is not None and self.use_community:
                    x = mark_tokens + community_tokens + time_embedding + positions
                else:
                    x = mark_tokens + time_embedding + positions
        
        # TRANSFORMER pass inputs and get outputs
        attn, dot = self.tblocks(x)
        
        if self.encoder_type == 'ATTN_RNN':
            h = self.rnn(attn, input.length)  # (b, t, e) or (b, t, 2e)  | e can be mark_emb_size or that + 1
        else:
            h = attn
        return h, dot  # (b, t, e) and (b, t, t)

        
    def positional_encoding(self, input):
        # DIM = (mark_embedding_size + 1) in_time value and in_marks embedding vector concatenated
        times = input.in_time
        time_tokens = times.unsqueeze(-1)
        if self.using_marks:
            mark_tokens = self.mark_embedding(input.in_mark)
            tokens = torch.cat([time_tokens, mark_tokens], -1)
        else:
            tokens = time_tokens
        b, t, e = tokens.size()  # batch, num of positions/steps in seq, inp embedded feat size
        
        # DIM = Position value in seq embedded to a size of (mark_embedding_size + 1) vector
        # positions = self.pos_embedding(torch.arange(t, device=self.device))[None, :, :].expand(b, t, e)
        positions = self.pos_encoding[None, :t, :].expand(b, t, e)
        
        # INPUTS: add or concat token(time.marks) with position_in_seq embedding
        x = torch.cat([tokens, positions], dim=2)  # x = (b, t, e)
        # x = tokens + positions
        return x
    
        
    def basis_time_encode(self, dt_attn_inputs):
        '''Mercer's time encoding '''
        N, max_len = dt_attn_inputs.size()
        expand_input = dt_attn_inputs[:, :, None].expand(N, max_len, self.time_dim).unsqueeze(-1)  # N, max_len, time_emb_dim, 1
        tmp = expand_input * (self.freq_var.to(self.device))
        sin_enc = torch.sin(tmp.to(self.device))
        cos_enc = torch.cos(tmp.to(self.device))
        time_enc = torch.cat([sin_enc, cos_enc], dim=-1) * (self.basis_expan_var.to(self.device))
        time_enc = time_enc.sum(dim=-1) + (self.basis_expan_var_bias.to(self.device))
        return time_enc

    
class Hypernet(nn.Module):
    """Hypernetwork for incorporating conditional information.

    Args:
        config: Model configuration. See `dpp.model.ModelConfig`.
        hidden_sizes: Sizes of the hidden layers. [] corresponds to a linear layer.
        param_sizes: Sizes of the output parameters.
        activation: Activation function.
    """
    def __init__(self, config, hidden_sizes=[], param_sizes=[1, 1], activation=nn.Tanh()):
        super().__init__()
        self.history_size = config.history_size
        self.embedding_size = config.embedding_size
        self.activation = activation

        # Indices for unpacking parameters
        ends = torch.cumsum(torch.tensor(param_sizes), dim=0)
        starts = torch.cat((torch.zeros(1).type_as(ends), ends[:-1]))
        self.param_slices = [slice(s.item(), e.item()) for s, e in zip(starts, ends)]

        self.output_size = sum(param_sizes)
        layer_sizes = list(hidden_sizes) + [self.output_size]
        # Bias used in the first linear layer
        self.first_bias = nn.Parameter(torch.empty(layer_sizes[0]).uniform_(-0.1, 0.1))
        if config.use_history:
            self.linear_rnn = nn.Linear(self.history_size, layer_sizes[0], bias=False)
        if config.use_embedding:
            self.linear_emb = nn.Linear(self.embedding_size, layer_sizes[0], bias=False)
        # Remaining linear layers
        self.linear_layers = nn.ModuleList()
        for idx, size in enumerate(layer_sizes[:-1]):
            self.linear_layers.append(nn.Linear(size, layer_sizes[idx + 1]))

    def reset_parameters(self):
        self.first_bias.data.fill_(0.0)
        if hasattr(self, 'linear_rnn'):
            self.linear_rnn.reset_parameters()
            nn.init.orthogonal_(self.linear_rnn.weight)
        if hasattr(self, 'linear_emb'):
            self.linear_emb.reset_parameters()
            nn.init.orthogonal_(self.linear_emb.weight)
        for layer in self.linear_layers:
            layer.reset_parameters()
            nn.init.orthogonal_(linear.weight)

    def forward(self, h=None, emb=None):
        """Generate model parameters from the embeddings.

        Args:
            h: History embedding, shape (*, history_size)
            emb: Sequence embedding, shape (*, embedding_size)

        Returns:
            params: Tuple of model parameters.
        """
        # Generate the output based on the input
        if h is None and emb is None:
            # If no history or emb are provided, return bias of the final layer
            # 0.0 is added to create a new node in the computational graph
            # in case the output will be modified by an inplace operation later
            if len(self.linear_layers) == 0:
                hidden = self.first_bias + 0.0
            else:
                hidden = self.linear_layers[-1].bias + 0.0
        else:
            hidden = self.first_bias
            if h is not None:
                hidden = hidden + self.linear_rnn(h)
            if emb is not None:
                hidden = hidden + self.linear_emb(emb)
            for layer in self.linear_layers:
                hidden = layer(self.activation(hidden))

        # Partition the output
        if len(self.param_slices) == 1:
            return hidden
        else:
            return tuple([hidden[..., s] for s in self.param_slices])