"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np
import cPickle

# data I/O
data = open('shakespeare_train.txt', 'r').read() # should be simple plain text file

# hyperparameters
hidden_size = 250 # size of hidden layer of neurons
seq_length = 25 # number of steps to unroll the RNN for
learning_rate = 1e-1

# model parameters
a = cPickle.load(open("char-rnn-snapshot.pkl"))
Wxh = a["Wxh"] 
Whh = a["Whh"]
Why = a["Why"]
bh = a["bh"]
by = a["by"]
chars, data_size, vocab_size, char_to_ix, ix_to_char = a["chars"].tolist(), a["data_size"].tolist(), a["vocab_size"].tolist(), a["char_to_ix"].tolist(), a["ix_to_char"].tolist()
# print len(chars), len(Wxh), len(Whh), len(Why), len(bh), len(by)
# print chars[9], Why[9]
# print '------'
# print by[9]
temperature = 1.2   # Setting up temperature
alpha = 1.0/temperature

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(alpha*y) / np.sum(np.exp(alpha*y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

def sample_text(sample_position):
  '''Part 1: This function generates a sample starting at "sample_position", 
  text length is adjustable in line 92.'''
  
  mWxh, mWhh, mWhy = a["mWxh"], a["mWhh"], a["mWhy"]
  mbh, mby = a["mbh"], a["mby"]
  # prepare inputs
  hprev = np.zeros((hidden_size,1)) # reset RNN memory
  inputs = [char_to_ix[ch] for ch in data[sample_position:sample_position+seq_length]]
  targets = [char_to_ix[ch] for ch in data[sample_position+1:sample_position+seq_length+1]]

  sample_ix = sample(hprev, inputs[0], 400)
  txt = ''.join(ix_to_char[ix] for ix in sample_ix)
  print '----\n %s \n----' % (txt, )
#sample_text(2000)

def complete_string(starter_index, sample_size):
  '''Part 2: This function completes a given starter string, 5 samples are
  generated each time, the starter string's index and sample size are 
  adjustable. The starter string plus the generated string are returned.'''
  
  mWxh, mWhh, mWhy = a["mWxh"], a["mWhh"], a["mWhy"]
  mbh, mby = a["mbh"], a["mby"]
  # prepare inputs
  hprev = np.zeros((hidden_size,1)) # reset RNN memory
  p = starter_index
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]

  # sample from the model after the given string with given size
  sample_ix = sample(hprev, inputs[-1], sample_size)
  txt = ''.join(ix_to_char[ix] for ix in sample_ix)
  print '----\n %s %s \n----' % (data[p:p+seq_length], txt)
complete_string(50000, 500)