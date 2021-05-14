
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from lang_dict import prepare_data, word_encodings_from_pairs, SOS_token
from luong_attention_batch import BatchEncoderRNN, BatchDecoderRNN
import time
import random


#check gpu device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device( "cpu")

#load data
language_file_path = "./fra.txt"
input_lang, output_lang, pairs = prepare_data(language_file_path, 'english', 'french')

### model parameters ###
hidden_size = 200
attn_method = 'general'
n_layers = 2
dropout = 0.05 

encoder = BatchEncoderRNN(input_lang.n_words, hidden_size, n_layers) 
decoder = BatchDecoderRNN(attn_method, hidden_size, output_lang.n_words, n_layers, dropout)
encoder.to(device)
decoder.to(device)

### training parameters ###
learning_rate = 0.0001
n_epochs = 200
clip = 5.0 #cap gradient
batch_size = 50


encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)


def masked_cross_entropy(prediction_padded, actual_padded, actual_lengths):
    # prediction_padded (seq len, batch, output size)
    # actual_padded (seq len, batch)
    # actual_lengths a list of length,size=batch size
    
    s1, b1, v1 = prediction_padded.shape
    s2, b2 = actual_padded.shape
    assert s1 == s2 #same sequence length
    
    #caculate the log likelihood for every prediction, padded or not
    log_prob = F.log_softmax(prediction_padded, dim = 2)
    
    #extend to the 3rd dimension for indicating the index of element
    #now log_prob and actual_padded are both 3d
    actual_padded = actual_padded.unsqueeze(2)
    
    #retrieve the score for the acutal elements
    #torch.gather gets the element at dim=2 and with index indicated by actual_padded
    #shape: (seq_len, batch size, 1)
    actual_scores = torch.gather(log_prob, 2, actual_padded)
    #squeeze the 3rd dimension, only 1 element now 
    #also swap the dimensions to (batch, seq len)
    actual_scores = actual_scores.squeeze(2).permute(1, 0)
    
    #change shape to (batch size, 1) for later masking
    actual_lengths = actual_lengths.unsqueeze(1)
    
    #create mask to exclude irrelevant scores
    #mask shape: (batch, seq len)
    mask = torch.arange(s2).expand(b2, s2) < actual_lengths
    mask = mask.to(device)
    
    #get rid of irrelevant scores
    #the irrelevant elements have 0 in the mask
    actual_scores = actual_scores * mask
    
    #get mean score
    mean_score = torch.sum(actual_scores) / torch.sum(actual_lengths)
    
    #negative log likelihood
    loss = - mean_score
    
    return loss


def train(input_seq_padded, input_seq_legnths, target_seq_padded, target_seq_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer):
    #train a batch of padded sequences
    #input_seq_padded & target_seq_padded : (seq len, batch size)
    #input_seq_legnths & target_seq_lengths : list of sequence lengths
    
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # pass input sequences through encoder
    encoder_init_hidden = encoder.init_hidden(batch_size)
    encoder_init_hidden = encoder_init_hidden.to(device)
    encoder_output, encoder_output_lengths, encoder_hidden = encoder(input_seq_padded, input_seq_legnths, encoder_init_hidden)
    
    # get input sequences for decoder
    # the first element is [SOS_token], the following (batchsize-1) elements are the target_seq[:-1]
    # shape the input to (seq_len, batch size)
    s, b = target_seq_padded.shape
    sos = torch.LongTensor([[SOS_token]]).expand((1, b))  
    sos = sos.to(device)
    decoder_input = torch.cat((sos, target_seq_padded[:-1, :]), 0) #append sos as head to every seq, remove the last element from every seq   

    # pass input sequence through decoder
    # always use teacher forcing 
    # use encoder_hidden as decoder's initial hidden
    # encoder_output has encoder hidden states at all time steps, shape: (seq_len, batch size, hidden size)
    # decoder_output shape: (batch size, target seq len, output size)
    decoder_output = decoder(decoder_input, target_seq_lengths, encoder_hidden, encoder_output, encoder_output_lengths)
    
    #calculate average loss
    loss = masked_cross_entropy(decoder_output, target_seq_padded, target_seq_lengths)

    # Backpropagation
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip) #clip the gradients to avoid explosion
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss.item()

loops = int(len(pairs)/batch_size)
for epoch in range(1, n_epochs + 1):
    epoch_loss = 0
    for i in range(1, loops+1):
        # get a batch of sequences' word encodings, plus EOS token at the end of each sequence
        # input words / output words is a list of lists
        # input_lengths / target_lengths is a list of sequence lengths
        inputs, input_lengths, targets, target_lengths = \
            word_encodings_from_pairs(input_lang, output_lang, random.choices(pairs, k = batch_size))  
        
        # pad the sequences
        # shape (max seq len, batch)
        inputs_padded = pad_sequence(inputs).to(device)
        targets_padded = pad_sequence(targets).to(device)
        input_lengths = torch.LongTensor(input_lengths)
        target_lengths = torch.LongTensor(target_lengths)
        
        # Run the train function
        loss = train(inputs_padded, input_lengths, targets_padded, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer)
        #print(loss)
        epoch_loss += loss
        
        if i%500 == 0:
            print('epoch {0} i {1} loss {2}'.format(epoch, i, epoch_loss/i))
            
        torch.cuda.empty_cache()      
        
    epoch_loss /= loops

    print("epoch {0} finished at {1}. loss is {2}".format(epoch, \
                                                          time.asctime(time.localtime(time.time())), \
                                                          epoch_loss))      



