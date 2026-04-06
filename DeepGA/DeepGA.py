# -*- coding: utf-8 -*-
""" Created on Sep 1 2024    @author: user
"""

from DeepGA.Operators import *
from DeepGA.EncodingClass import Encoding
from DeepGA.Decoding import *
from DeepGA.DistributedTraining import *
from torch.utils.data import DataLoader
import timeit
import torch
import pickle
import torch.nn as nn
import os
from pathlib import Path
import pandas as pd
import copy

def deepGA(execution: int, memoryC: bool, train_epochs: int, train_dl:DataLoader, val_dl:DataLoader,lr: float,
           min_conv: int, max_conv: int, min_full: int, max_full: int, max_params: int, cr: float, mr : float,  
           N: int, T: int, t_size: int, w: float, device: torch.device, chck_dir: str, n_channels =  int , n_classes=int, out_size = int, loss_func=None):
   
  num_epochs = train_epochs #Epochs to train each individual during the GA
  #Defining loss function
  #loss_func = nn.CrossEntropyLoss()
  #Reading GPU
  device1 = device

  # Indicate path for checkpoint
  if not os.path.exists(chck_dir):  
      os.makedirs(chck_dir)      

  '''Initialize population'''
  # Check if checkpoint is available
  chkpoint_obj = Path(chck_dir + str(execution) + "_checkpoint.pkl")
  if chkpoint_obj.exists():
    print("Re-Initialize population")
    with open(chck_dir + str(execution) + "_checkpoint.pkl", "rb") as p:
      values = pickle.load(p)
    start = timeit.default_timer() - values['time']
    pop = values['pop']
    bestAcc = values['bestAcc']
    bestF = values['bestF']
    bestParams = values['bestParams']
    t = values['t']
    if t == T:
      print('The maximum number of generations has been reached. Please run a new execution.')
      leader = max(pop, key = lambda x: x[1])
    evals = values['evals']
    cacheM = values['cacheM']
    meanfitpop = values['meanfitpop']
    meanAccpop = values['meanAccpop']
    meanParpop = values['meanParpop']
  else:
    print('Initialize population')
    start = timeit.default_timer()
    pop = []
    bestAcc = []
    bestF = []
    bestParams = []
    t = 0  # Generaciones
    evals = 0 # Evaluaciones
    cacheM = {}
    meanfitpop = []
    meanAccpop = []
    meanParpop = []

    while len(pop) < N:
      #acc_list = manager.list()
      acc_list = []

      #Creating genomes (genetic encoding)
      e1 = Encoding(min_conv, max_conv, min_full, max_full)

      if memoryC:
        strIDe1 = str([e1.n_conv, e1.n_full, e1.first_level, e1.second_level])
        if strIDe1 in cacheM:
          fit1 = cacheM[strIDe1][0]
          acc1 = cacheM[strIDe1][1]
          pars1 = cacheM[strIDe1][2]
        else:
          #Decoding the networks
          network1 = decoding(e1, n_channels, out_size, n_classes)
          #Creating the CNNs
          cnn1 = CNN(e1, network1[0], network1[1], network1[2])
          #Evaluate individuals
          fit1, acc1, pars1, _ = training('1', device1, cnn1, num_epochs, loss_func, train_dl, val_dl, lr, w, max_params, acc_list)
          #Store fitness in memory
          cacheM[strIDe1] = [fit1, acc1, pars1]
          evals += 1

      else:
        #Decoding the networks
        network1 = decoding(e1, n_channels, out_size, n_classes)

        #Creating the CNNs
        cnn1 = CNN(e1, network1[0], network1[1], network1[2])

        #Evaluate individuals
        fit1, acc1, pars1, _ = training('1', device1, cnn1, num_epochs, loss_func, train_dl, val_dl, lr, w, max_params, acc_list)
        evals += 1

      print(fit1, acc1, pars1)
      pop.append([e1, fit1, acc1, pars1])


  '''Genetic Algorithm'''
  print('--------------------------------------------')
  while t < T:
    print('Generation: ', t)

    #Parents Selection
    parents = []
    while len(parents) < int(N/2):
      #Tournament Selection
      tournament = random.sample(pop, t_size)
      p1 = selection(tournament, 'max')
      tournament = random.sample(pop, t_size)
      p2 = selection(tournament, 'max')
      while p1 == p2:
        tournament = random.sample(pop, t_size)
        p2 = selection(tournament, 'max')

      parents.append(p1)
      parents.append(p2)

    #Reproduction
    offspring = []
    iter_parents = 0
    while len(offspring) < int(N/2):
      #par = random.sample(parents, 2)
      #Crossover + Mutation
      p1 = parents[iter_parents][0]
      p2 = parents[iter_parents + 1][0]
      if cr >= random.uniform(0,1): #Crossover
        c1, c2 = crossover(p1, p2)
      else:
        c1 = deepcopy(p1)
        c2 = deepcopy(p2)

      #Mutation
      if mr >= random.uniform(0,1):
        mutation(c1)

      if mr >= random.uniform(0,1):
        mutation(c2)

      #Evaluate offspring
      acc_list = []

      if memoryC:
        strIDc1 = str([c1.n_conv, c1.n_full, c1.first_level, c1.second_level])
        strIDc2 = str([c2.n_conv, c2.n_full, c2.first_level, c2.second_level])

        if strIDc1 in cacheM:
          fit1 = cacheM[strIDc1][0]
          acc1 = cacheM[strIDc1][1]
          pars1 = cacheM[strIDc1][2]
          print(fit1, acc1, pars1)
        else:
          network1 = decoding(c1, n_channels, out_size, n_classes)
          #Creating the CNN
          cnn1 = CNN(c1, network1[0], network1[1], network1[2])
          #Evaluate individuals
          fit1, acc1, pars1, _ = training('1', device1, cnn1, num_epochs, loss_func, train_dl, val_dl, lr, w, max_params, acc_list)
          #Store fitness in memory
          cacheM[strIDc1] = [fit1, acc1, pars1]
          print(fit1, acc1, pars1)
          evals += 1

        if strIDc2 in cacheM:
          fit2 = cacheM[strIDc2][0]
          acc2 = cacheM[strIDc2][1]
          pars2 = cacheM[strIDc2][2]
          print(fit2, acc2, pars2)
        else:
          #Decoding the network
          network2 = decoding(c2, n_channels, out_size, n_classes)
          #Creating the CNN
          cnn2 = CNN(c2, network2[0], network2[1], network2[2])
          #Evaluate individuals
          fit2, acc2, pars2, _ = training('1', device1, cnn2, num_epochs, loss_func, train_dl, val_dl, lr, w, max_params, acc_list)
          #Store fitness in memory
          cacheM[strIDc2] = [fit2, acc2, pars2]
          print(fit2, acc2, pars2)
          evals += 1
      else:
        #Decoding the network
        network1 = decoding(c1, n_channels, out_size, n_classes)
        network2 = decoding(c2, n_channels, out_size, n_classes)

        #Creating the CNN
        cnn1 = CNN(c1, network1[0], network1[1], network1[2])
        cnn2 = CNN(c2, network2[0], network2[1], network2[2])

        #Evaluate individuals
        fit1, acc1, pars1, _ = training('1', device1, cnn1, num_epochs, loss_func, train_dl, val_dl, lr, w, max_params, acc_list)
        print(fit1, acc1, pars1)
        evals += 1

        fit2, acc2, pars2, _ = training('1', device1, cnn2, num_epochs, loss_func, train_dl, val_dl, lr, w, max_params, acc_list)
        print(fit2, acc2, pars2)
        evals += 1

      offspring.append([c1, fit1, acc1, pars1])
      offspring.append([c2, fit2, acc2, pars2])

      iter_parents += 2

    #Replacement with elitism
    pop = pop + offspring
    pop.sort(reverse = True, key = lambda x: x[1])
    pop = pop[:N]

    leader = max(pop, key = lambda x: x[1])
    bestAcc.append(leader[2])
    bestF.append(leader[1])
    bestParams.append(leader[3])
    meanfitpop.append(sum([q[1] for q in pop])/N)
    meanAccpop.append(sum([q[2] for q in pop])/N)
    meanParpop.append(sum([q[3] for q in pop])/N)

    t += 1

    # Making checkpoint
    time = timeit.default_timer() - start
    current_state: dict = dict(pop=pop, bestAcc=bestAcc,bestF=bestF,
                               bestParams=bestParams, t=t, evals=evals,
                               time=time, cacheM=cacheM, meanfitpop=meanfitpop,
                               meanAccpop=meanAccpop, meanParpop=meanParpop)
    with open(chck_dir + str(execution) + "_checkpoint.pkl", "wb") as p:
      pickle.dump(current_state, p)

    print('Best fitness: ', leader[1])
    print('Best accuracy: ', leader[2])
    print('Best No. of Params: ', leader[3])
    print('No. of Conv. Layers: ', leader[0].n_conv)
    print('No. of FC Layers: ', leader[0].n_full)
    print('--------------------------------------------')

  bestind = copy.deepcopy(leader)
  results = pd.DataFrame(list(zip(bestAcc, bestF, bestParams, meanfitpop,  meanAccpop, meanParpop)),
                         columns = ['Accuracy', 'Fitness', 'No. Params', 'MeanFit', 'MeanAcc', 'MeanPar'])
  print(results)

  return results, pop, bestind  


def final_evaluation(execution: int, bestind: list, train_dl: DataLoader, val_dl: DataLoader, lr: float,
                     max_params: int, w: float, device: torch.device, train_epochs: int, loss_func, chck_dir: str,
                     n_channels =  int , n_classes=int, out_size = int):
  
  chkpoint_obj = Path(chck_dir + "Model_Exec_"+str(execution)+"_Epoch_"+str(train_epochs)+"_point.pkl")
  if not chkpoint_obj.exists():
     print("Training final model from best individual") 
     start = timeit.default_timer()
     #Decoding the networks of the best individual
     network1 = decoding(bestind[0], n_channels, out_size, n_classes)
     acc_list = []
     #Creating the CNNs
     cnnfin = CNN(bestind[0], network1[0], network1[1], network1[2])
     #Evaluate individuals
     fitfin, accfin, parsfin, CNNModel = training('1', device, cnnfin, train_epochs, loss_func, train_dl, val_dl, lr, w, max_params, acc_list)
     print(fitfin, accfin, parsfin)
     stop = timeit.default_timer()
     execution_timeS = (stop-start)
     execution_timeH = execution_timeS/3600
     print("Execution time: ", execution_timeS, " seconds")
     print("Execution time: ", execution_timeH, " hours")
     print("Accuracy: ", accfin)
     CNNModel.to("cpu")
     current_state_model: dict = dict(modelo=CNNModel)
     with open(chkpoint_obj, "wb") as p:
         pickle.dump(current_state_model, p)

  else: 
     print("Loading final model")
     with open(chkpoint_obj, "rb") as p:
        values = pickle.load(p)
     CNNModel = values['modelo']
   
  return CNNModel