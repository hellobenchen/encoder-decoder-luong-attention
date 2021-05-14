
"""
The batch version of encoder-decoder model with Luong attention - general method
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BatchEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers = 1, n_directions = 1):
        
        # the batch version of encoder
        # use pack_padded_sequence as input
        # input_size: size of the word encoding, i.e. size of the vocabulary, or # of rows in the embedding matrix
        # hidden_size: the embedding vector's size, and also the GRU's hidden states size here
        # n_layers: # of layers of GRU
        
        super(BatchEncoderRNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = n_directions
        
        # embedding layer, simply a look up table/matrix
        # use pre-trained embedding later
        # embedding input: (*)
        # embedding output (*, hidden size)
        self.embedding = nn.Embedding(input_size, hidden_size) 
        
        #the rnn unit. set rnn's hidden size the same as the embedding's hidden size. no need to be the same though.
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                m.weight.data.uniform_(-1, 1) 
        
    def forward(self, padded_sequences, seq_lengths, initial_hidden_states):
        # sequences are padded to the same length
        # padded_sequences: (seq len, batch size), each word in the sequence is represented by its index, i.e. a single number 
        # seq_lengths: list of sequence lengths
        # initial_hidden_states is from the init_hidden function
        
        #get word embedding, shape: (seq_len, batch size, hidden size)
        #the output embedding is padded as well, because the input is padded
        embedded = self.embedding(padded_sequences)
        
        #pack the padded embedding
        #set enforce_sorted = false, otherwise need to sort by sequence length descendingly
        packed = pack_padded_sequence(embedded, seq_lengths, enforce_sorted=False)
        packed.to(device)
        
        # feed packed embedding sequences through gru directly
        # rnn unit is optimized to handle packed sequences
        packed_output, last_hidden = self.gru(packed, initial_hidden_states) 
        
        # retore the output to padded format
        # output: only the last layer's hidden states at all time steps  
        #         shape = (seq_len, batch, num_directions * hidden_size), num_directions = 1 here
        # hidden: the hidden states of all layers at the last time step t
        #         shape = (num_layers * num_directions, batch, hidden_size)
        padded_output, output_lens = pad_packed_sequence(packed_output)                 
        
        return padded_output, output_lens, last_hidden

    def init_hidden(self, batch_size):
        #gru init hidden states shape: (num_layers * num_directions, batch, hidden_size)
        return torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size) 



class BatchAttention(nn.Module):
    def __init__(self, method, hidden_size):
        # implement Luong's attention mechanism
        # hidden_size: size of GRU's hidden states
        
        super(BatchAttention, self).__init__()
        #assert(method in ['dot', 'general', 'concat']) #only the three methods
        assert(method in ['general']) #only general
        self.method = method
        self.hidden_size = hidden_size
        
        #methods for calcualting the energy between target hidden states and source hidden states
        if self.method == 'general':
            # i.e. target hidden states <dot > Weights <dot> source hidden states
            # the weights are simualted by a fully connected linear layer
            # encoder's output, i.e. source hidden states at all time steps, go through a linear layer
            # and dot with decoder's current hidden states in the forward method
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0)


    def forward(self, target_hidden_padded, target_hidden_lengths, source_hidden_padded, source_hidden_lengths):
         # target_hidden_padded shape: (seq_len, batch, hidden size)
         # source_hidden_padded shape: (seq_len, batch, hidden size)
         # target_hidden_lengths and source_hidden_lengths: a list of seq lengths
        
        # Calculate energies between target hidden states and source hidden states
        # energy shape : (batch size, target seq len, source seq len)
        energy_padded = self.score(target_hidden_padded, source_hidden_padded)

        # the following applies mask on the energy_padded to zero out those padded elements
        # get the target and source shapes first 
        s1, b1, h1 = target_hidden_padded.shape
        s2, b2, h2 = source_hidden_padded.shape 
        
        #add an extra dimension for creating mask later, i.e. 1d to 2d
        target_hidden_lengths = target_hidden_lengths.unsqueeze(1)
        source_hidden_lengths = source_hidden_lengths.unsqueeze(1)
               
        #get the 2d mask for target and source
        #only the elements within the actual sequence length is set to 1 (true), others are set to 0 (false)
        # e.g. a 5-element sequence with actual length 3 will have [1,1,1,0,0] so as to zero out energy of the 4th and 5th elements.
        target_mask = torch.arange(s1).expand(b1, s1) < target_hidden_lengths       
        source_mask = torch.arange(s2).expand(b2, s2) < source_hidden_lengths
        
        # combine the target and source masks, 2d to 3d
        # get the mask, shape(batch, s1, s2) which matches the energy's shape
        target_mask = target_mask.unsqueeze(2).expand(b1, s1, s2)
        source_mask = source_mask.unsqueeze(2).expand(b2, s2, s1).permute(0, 2, 1) #shape (b2, s1, s2)
        
        # only mask element that is 1 in both target and source masks remains 1
        mask = target_mask * source_mask
        mask = mask.to(device)
       
        # energy_padded and mask shape (batch, s1, s2)
        # calculate the weight of each source hidden states for each target hidden states
        # shape (batch, s1, s2), weight is a probability distribution, irrelevant weight element is 0
        attn_weights_padded = self.masked_softmax(energy_padded, mask) 
               
        return attn_weights_padded

        
    def score(self, target_hidden, source_hidden):
         # Calculate the energy between source hidden states and target hidden states
         # Those padded states will be calculated as well. A later step will mask them out.
         # target_hidden shape: (seq_len, batch size, hidden size)
         # source_hidden shape: (seq_len, batch size, hidden size)
         
        s1, b1, h1 = target_hidden.shape
        s2, b2, h2 = source_hidden.shape   
        assert b1 == b2 and h1 == h2  # same batch size, same hidden size, but target and source can have different sequence lengths
                     
        if self.method == 'general':
            #target hidden <dot > weights <dot> source hidden
            
            # reshape into a bigger batch of size s2 * b2 to feed through linear layer
            source_hidden = source_hidden.view(s2 * b2, h2)
        
            # swap batch and seq_len dimension for later bmm operation, new shape: (b1, s1, h1)
            target_hidden = target_hidden.permute(1, 0, 2)
            
            #go through linear layer, i.e. weights <dot> source hidden
            #output shape is (s2 * b2, h2)
            energy = self.attn(source_hidden) 
            
            #reshape energy into (b2, h2, s2) for later bmm operation
            energy = energy.view(s2, b2, h2).permute(1, 2, 0)
            
            #calculate energy between every target hidden states and every source hidden states
            #matrix batch (b1, s1, h1) <bmm> matrix batch (b2, h2, s2)
            #yielding a output shape of (b1, s1, s2), i.e. (batch_size, target seq len, source target len)
            energy = target_hidden.bmm(energy)
            
            return energy
           
    def masked_softmax(self, energy, mask): 
        # energy: (batch size, target seq len, source seq len)
        # mask: (batch size, target seq len, source seq len)
        M = torch.max(energy)
        
        exps = torch.exp(energy - M) #avoid overflow
        
        #only valid target-source hidden pairs keep the exp value, others are 0
        masked_exps = exps * mask.float() 
        
        #sum over dim 2, i.e. the source hidden states
        masked_sums = masked_exps.sum(dim = 2, keepdim = True) + 1e-5  
        
        # exp(i)/sum(exp(j)), i.e. softmax for each valid energy element
        prob = masked_exps/masked_sums
        
        return prob
    

class BatchDecoderRNN(nn.Module):
    def __init__(self, attn_method, hidden_size, output_size, n_layers=1, dropout=0.1, embedding_bin_file=None):
        # the decoder uses Luong's attention
        # attention method can be 'dot', 'general' or 'concat', only general is implemented here
        # hidden size is the internal hidden states' size
        # output size is the vocabulary size
        # n layers of rnn units
        # dropout is applied on the hidden states output when passed onto the next layer, i.e. hidden * dropout
               
        super(BatchDecoderRNN, self).__init__()
        self.attn_method = attn_method
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        #get input sequences' embedding
        #can use a pre-trained embedding here
        if embedding_bin_file is None:
            self.embedding = nn.Embedding(output_size, hidden_size)
        # else:
        #     model = gensim.models.fasttext.load_facebook_vectors(embedding_bin_file)
        #     self.embedding = nn.Embedding.from_pretrained(model.wv) #model.wv is the weights matrix
    
        #the rnn unit
        #dropout is applied only if more than 1 layers
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        
        #calculate attention weights
        self.attn = BatchAttention(attn_method, hidden_size)
        
        #feed hidden + context through a linear layer to get the attentional hidden states
        self.linear = nn.Linear(hidden_size * 2, hidden_size)
        
        #output attentional hidden states to final out vector(vocabulary size)
        self.out = nn.Linear(hidden_size, output_size)
        
        for m in self.modules():
            if isinstance(m, nn.Embedding):
                #print('init decoder embedding')
                m.weight.data.uniform_(-1, 1)
                
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0)      

    def forward(self, target_seq_padded, target_seq_lengths, initial_hidden, source_hidden_padded, source_hidden_lengths):
        # target_seq_padded shape: (seq_len, batch), every word is represented by its index
        # target_seq_lengths: list of actual sequence lengths
        # initial_hidden: hidden states from previous batch, shape:(num_layers * num_directions, batch, hidden_size), here num_directions=1
        # source_hidden_padded: (seq_len, batch, hidden size)
        # source_hidden_lengths: list of actual sequence lengths
        
        # get embedding, (seq_len, batch) --> (seq_len, batch, hidden size)
        embedded = self.embedding(target_seq_padded)
        
        #pack the embedded
        packed = pack_padded_sequence(embedded, target_seq_lengths, enforce_sorted=False)

        #run the packed batch through rnn unit
        packed_output, _ = self.gru(packed, initial_hidden) 
        
        # restore the output and hidden to padded format
        # padded_output shape (seq_len, batch size, hidden size)
        # output_lens : a list of lengths
        padded_output, output_lens = pad_packed_sequence(packed_output)
        
        # the padded_output shape (seq_len, batch size, hidden size)
        # get attention weights of source hidden states for every target hidden states
        # attn_weights_padded shape: (batch, target seq len, source seq len), padded with 0
        attn_weights_padded = self.attn(padded_output, output_lens, source_hidden_padded, source_hidden_lengths) 

        # source_hidden_padded shape:(source seq len, batch, hidden size)
        # swap dim 0 and dim 1 to (batch, source seq len, hidden size)
        source_hidden_padded = source_hidden_padded.permute(1, 0, 2)
        
        # calcuate weighted average of all source hidden states for every target hidden states
        # the bmm outputs shape: (batch, target seq len, hidden size)
        # the irrelevant time steps (padded with 0 weight) contributes 0 to the weighted average 
        # context shape: (batch, target seq len, hidden size)
        context = attn_weights_padded.bmm(source_hidden_padded)

        # concat target hidden states with context
        # this outputs a shape of (batch, target seq len, hidden size * 2)
        padded_output = padded_output.permute(1,0,2)
        concat = torch.cat((padded_output, context), 2)
        
        #combine dim 0 and 1 so as to feed concat through a linear layer
        b, s, h = concat.shape
        concat = concat.view( b * s, h)
        
        # run the concat (b * s, hidden size * 2) through linear layer, 
        # and activate it to get attentional hidden states
        # output shape ( b * s, hidden size)
        attentional_hidden = torch.tanh(self.linear(concat))
        
        # predict the next word using attentional_hidden, through the last linear layer
        # apply softmax to get probability over a vocabulary size vector
        # output shape (b * s, output size)
        output = self.out(attentional_hidden)
        
        # retore output back to shape (target seq len, batch, output size)
        output = output.view(b, s, -1).permute(1, 0, 2)
        
        return output
