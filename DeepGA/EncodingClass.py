# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 09:57:16 2020

@author: user
"""
import random

'''Hyperparameters configuration'''
#Convolutional layers
FSIZES = [3, 5, 7, 9] # Odd Sizes Are Preferred
NFILTERS = [8, 16, 32, 64, 128, 256]

#Pooling layers
PSIZES = [2,3] #[2,3,4,5]
PTYPE = ['max', 'avg']

#Fully connected layers
NEURONS = [16, 32, 64, 128, 256] # for big layers

class Encoding:
  def __init__(self, minC, maxC, minF, maxF):
    self.n_conv = random.randint(minC, maxC)
    self.n_full = random.randint(minF, maxF)


    '''First level encoding'''
    self.first_level = []

    #Feature extraction part
    for i in range(self.n_conv):
      layer = {'type' : 'conv',
                'nfilters' : random.choice(NFILTERS),
                'fsize' : random.choice(FSIZES),
                'pool' : random.choice(['max', 'avg', 'off']),
                'psize' : random.choice(PSIZES)
              }
      self.first_level.append(layer)

    #Fully connected part
    for i in range(self.n_full):
      layer = {'type' : 'fc',
                'neurons' : random.choice(NEURONS)}

      self.first_level.append(layer)


    '''Second level encoding'''
    self.second_level = []
    prev = -1
    for i in range(self.n_conv):
      if prev < 1:
        prev += 1
      if prev >= 1:
        for _ in range(prev-1):
            self.second_level.append(random.choice([0,1]))
        prev += 1

   