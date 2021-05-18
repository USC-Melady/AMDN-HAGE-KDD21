import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dpp
from dpp.data import Batch

from dpp.utils import DotDict
from dpp.nn import BaseModule


class Model(BaseModule):
    """Base model class.

    Attributes:
        rnn: RNN for encoding the event history.
        embedding: Retrieve static embedding for each sequence.
        decoder: Compute log-likelihood of the inter-event times given hist and emb.

    Args:
        config: General model configuration (see dpp.model.ModelConfig).
        decoder: Model for computing log probability of t given history and embeddings.
            (see dpp.decoders for a list of possible choices)
    """
    def __init__(self, config, decoder):
        super().__init__()
        self.device = config.device
        self.use_history(config.use_history)
        self.use_embedding(config.use_embedding)
        self.use_marks(config.use_marks)

        self.encoder_type = config.encoder_type
        if self.encoder_type == 'RNN':
            self.rnn = dpp.nn.RNNLayer(config)
        else:
            self.rnn = dpp.nn.AttentiveLayer(config)
        # attend on different history when predicting time (may be popular post)
        # but attend on different events (following) when predicting marks
        
        # Sequence embedding. nothing to do with marks embedding
        if self.using_embedding:
            self.embedding = nn.Embedding(config.num_embeddings, config.embedding_size)
            self.embedding.weight.data.fill_(0.0)

        if self.using_marks:
            self.num_classes = config.num_classes
            self.mark_layer = nn.Sequential(
                nn.Linear(config.history_size, config.history_size),
                nn.ReLU(),
                nn.Linear(config.history_size, self.num_classes)
            )  # output for mark predictions (not embedding of in_marks)
        self.decoder = decoder
        
    
    def forward(self, in_time, out_time, length, index, in_mark, out_mark, use_marks, device):
        input = Batch(in_time, out_time, length, index, in_mark, out_mark)
        if use_marks:
            log_prob, mark_nll, accuracy = self.log_prob(input)
            loss = -self.aggregate(log_prob, input.length, device) + self.aggregate(mark_nll, input.length, device)
            del log_prob, mark_nll, accuracy
        else:
            loss = -self.aggregate(self.log_prob(input), input.length, device)
        return loss


    def mark_nll(self, h, y):
        """Compute log likelihood and accuracy of predicted marks

        Args:
            h: History vector
            y: Out marks, true label

        Returns:
            loss: Negative log-likelihood for marks, shape (batch_size, seq_len)
            accuracy: Percentage of correctly classified marks
        """
        x = self.mark_layer(h)
        x = F.log_softmax(x, dim=-1)
        loss = F.nll_loss(x.view(-1, self.num_classes), y.view(-1), reduction='none').view_as(y)
        accuracy = (y == x.argmax(-1)).float()
        
        return loss, accuracy

    def log_prob(self, input):
        """Compute log likelihood of the inter-event timesi in the batch.

        Args:
            input: Batch of data to score. See dpp.data.Input.

        Returns:
            time_log_prob: Log likelihood of each data point, shape (batch_size, seq_len)
            mark_nll: Negative log likelihood of marks, if using_marks is True
            accuracy: Accuracy of marks, if using_marks is True
        """
        
        # Encode the history with an RNN
        if self.using_history:
            if self.encoder_type == 'RNN':
                h = self.rnn(input)
            else:
                h, _ = self.rnn(input)  # has shape (batch_size, seq_len, rnn_hidden_size)
        else:
            h = None
        # Get sequence embedding
        if self.using_embedding:
            # has shape (batch_size, seq_len, embedding_size)
            emb = self.embedding(input.index).unsqueeze(1).repeat(1, input.out_time.shape[1], 1)
        else:
            emb = None

        t = input.out_time  # has shape (batch_size, seq_len)
        time_log_prob = self.decoder.log_prob(t, h, emb)

        if self.using_marks:
            mark_nll, accuracy = self.mark_nll(h, input.out_mark)
            return time_log_prob, mark_nll, accuracy

        return time_log_prob

    
    def aggregate(self, values, lengths, device):
        """Calculate masked average of values.

        Sequences may have different lengths, so it's necessary to exclude
        the masked values in the padded sequence when computing the average.

        Arguments:
            values (list[tensor]): List of batches where each batch contains
                padded values, shape (batch size, sequence length)
            lengths (list[tensor]): List of batches where each batch contains
                lengths of sequences in a batch, shape (batch size)

        Returns:
            mean (float): Average value in values taking padding into account
        """

        if not isinstance(values, list):
            values = [values]
        if not isinstance(lengths, list):
            lengths = [lengths]

        total = 0.0
        for batch, length in zip(values, lengths):
            length = length.long()
            mask = torch.arange(batch.shape[1], device=device)[None, :] < length[:, None]
            mask = mask.float()

            batch[torch.isnan(batch)] = 0 # set NaNs to 0
            batch *= mask

            total += batch.sum()

        total_length = sum([x.sum() for x in lengths])

        return total / total_length
    
    
    def attention_weights(self, input, device):
        if self.encoder_type == 'RNN':
            return None, None
        # Encode the history with an RNN
        h, dot = self.rnn(input)  # dot has shape (batch_size, seq_len, seq_len)
        
        A = torch.zeros((self.num_classes, self.num_classes))
        counts = torch.zeros((self.num_classes, self.num_classes))
        
        for seq_attn, marks, length in zip(dot, input.in_mark, input.length):
            length = length.long()
            marks = marks[:length]
            rows = marks.unsqueeze(1).repeat(1, length)
            cols = rows.transpose(1, 0)
            A[rows, cols] += seq_attn[:length, :length].detach().cpu()
            counts[rows, cols] += (seq_attn[:length, :length] > 0).detach().cpu().float()
        
        return A.detach().cpu(), counts.detach().cpu()
        
class EnhancedModel(Model):
    
    def __init__(self, config, decoder, community_num):
        super().__init__(config, decoder)
        

    def forward(self, in_time, out_time, length, index, in_mark, out_mark, use_marks, device, gmm_prob = None):
        input = Batch(in_time, out_time, length, index, in_mark, out_mark)
        if use_marks:
            log_prob, mark_nll, accuracy = self.log_prob(input, gmm_prob)
            loss = -self.aggregate(log_prob, input.length, device) + self.aggregate(mark_nll, input.length, device)
            del log_prob, mark_nll, accuracy
        else:
            loss = -self.aggregate(self.log_prob(input, gmm_prob), input.length, device)
        return loss
    
    def log_prob(self, input, gmm_prob = None):
        """Compute log likelihood of the inter-event timesi in the batch.

        Args:
            input: Batch of data to score. See dpp.data.Input.

        Returns:
            time_log_prob: Log likelihood of each data point, shape (batch_size, seq_len)
            mark_nll: Negative log likelihood of marks, if using_marks is True
            accuracy: Accuracy of marks, if using_marks is True
        """
        
        # Encode the history with an RNN
        if self.using_history:
            if self.encoder_type == 'RNN':
                h = self.rnn(input)
            else:
                h, _ = self.rnn(input, gmm_prob)  # has shape (batch_size, seq_len, rnn_hidden_size)
        else:
            h = None
        # Get sequence embedding
        if self.using_embedding:
            # has shape (batch_size, seq_len, embedding_size)
            emb = self.embedding(input.index).unsqueeze(1).repeat(1, input.out_time.shape[1], 1)
        else:
            emb = None

        t = input.out_time  # has shape (batch_size, seq_len)
        time_log_prob = self.decoder.log_prob(t, h, emb)

        if self.using_marks:
            mark_nll, accuracy = self.mark_nll(h, input.out_mark)
            return time_log_prob, mark_nll, accuracy

        return time_log_prob

class ModelConfig(DotDict):
    """Configuration of the model.
    This config only contains parameters that need to be know by all the
    submodules. Submodule-specific parameters are passed to the respective
    constructors.
    """
    def __init__(self,
                 encoder_type='ATTN',
                 use_history=True,
                 history_size=None,
                 rnn_type='RNN',
                 use_embedding=False,
                 embedding_size=32,
                 num_embeddings=None,
                 use_marks=False,
                 mark_embedding_size=64,
                 num_classes=None,
                 heads=None,
                 depth=None,
                 wide=None,
                 seq_length=None,
                 device=None,
                 pos_enc=False,
                 add=0,
                 time_opt='delta',
                 expand_dim=10,
                 gmm_k=2,
                 use_community = False):
        super().__init__()
        
        # Encoder type and parameters
        self.encoder_type = encoder_type  # {'RNN', 'ATTN', 'ATTN_RNN'}  
        # RNN is original LogNormMix paper, ATTN is amdn paper with last history state taken as context encoding of history
        # ATTN_RNN is amdn paper with RNN last state from outputs of each attn output history state taken as context encoding of history
        self.use_history = use_history  # True
        self.history_size = history_size   # None if ATTN or ATTN_RNN is used, because the size depends on ATTN block (updated below)
        self.rnn_type = rnn_type  # {'RNN', 'LSTM', 'GRU'}
        
        # Marks embeddings parameters
        self.use_marks = use_marks  # True
        self.mark_embedding_size = mark_embedding_size  # 32|64
        self.num_classes = num_classes  # process_dim
        
        # Attentive parameters
        self.heads = heads
        self.depth = depth
        self.wide = wide
        self.seq_length = seq_length  # 128
        
        self.pos_enc = pos_enc  # True if only position encoded, not time; False, pos and time encoding
        self.expand_dim = expand_dim  # range 5|30 for time encoding kernel frequency
        self.add = add
        self.time_opt = time_opt
        
        if self.encoder_type in set(['ATTN', 'ATTN_RNN']) and self.pos_enc:
            self.history_size = 2 * (mark_embedding_size + 1)  # if posemb concat (marks.time)
            if self.add:
                self.history_size = mark_embedding_size + 1
        else:
            if use_community:
                self.history_size = 4 * mark_embedding_size   # if posemb and community emb concat marks (time func encoding)
            else:
                self.history_size = 3 * mark_embedding_size   # if posemb concat marks (time func encoding)
            if self.add:
                self.history_size = mark_embedding_size
        self.device = device
        
        # Sequence embedding parameters
        self.use_embedding = use_embedding  # False
        self.embedding_size = embedding_size  # None
        if use_embedding and num_embeddings is None:
            raise ValueError("Number of embeddings has to be specified")
        self.num_embeddings = num_embeddings  # None
        self.gmm_k = gmm_k
        self.use_community = use_community