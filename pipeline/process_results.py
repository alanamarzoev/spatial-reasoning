import os, torch, sys, subprocess, pickle
import matplotlib.pyplot as plt
import numpy as np 
import data
import pipeline
import environment 
import argparse
   

def run_eval(pickle_path):  
    metric_path = pickle_path 
    quality_path = os.path.join( metric_path, 'quality')
    if not os.path.exists( quality_path ):
        subprocess.call(['mkdir', quality_path])

    save_path = pickle_path 
   
    predictions = pickle.load( open(os.path.join(save_path, 'test_predictions.p'), 'rb') ).squeeze()
    targets = pickle.load( open(os.path.join(save_path, 'test_targets.p'), 'rb') ).squeeze()
    rewards = pickle.load( open(os.path.join(save_path, 'test_rewards.p'), 'rb') ).squeeze()
    terminal = pickle.load( open(os.path.join(save_path, 'test_terminal.p'), 'rb') ).squeeze()

    rewards = rewards.cpu().numpy()
    terminal = terminal.cpu().numpy()

    num_worlds = targets.shape[0]
    print 'Num worlds: {}'.format(num_worlds)

    mse = np.sum(np.power(predictions - targets, 2)) / predictions.size
    print 'MSE: {}'.format(mse)

    cumulative_normed = 0
    manhattan = 0
    cumulative_per_score = 0
    cumulative_score = 0

    for ind in range(num_worlds):
        pred = predictions[ind]
        targ = targets[ind]

        pred_max = np.unravel_index(np.argmax(pred), pred.shape)
        targ_max = np.unravel_index(np.argmax(targ), targ.shape)
        man = abs(pred_max[0] - targ_max[0]) + abs(pred_max[1] - targ_max[1])

        unif = np.ones( pred.shape )
        rew = rewards[ind]
        term = terminal[ind]

        mdp = environment.MDP(None, rew, term)
        si = pipeline.ScoreIteration(mdp, pred)
        avg_pred, scores_pred = si.iterate()

        mdp = environment.MDP(None, rew, term)
        si = pipeline.ScoreIteration(mdp, targ)
        avg_targ, scores_targ = si.iterate()

        mdp = environment.MDP(None, rew, term)
        si = pipeline.ScoreIteration(mdp, unif)
        avg_unif, scores_unif = si.iterate()
     
        start_pos = (np.random.randint(10), np.random.randint(10))
        score = simulate_single(rew, term, pred, start_pos)

        normed = (avg_pred - avg_unif) / (avg_targ - avg_unif)
        cumulative_normed += normed
        manhattan += man
        cumulative_score += score

    avg_normed = float(cumulative_normed) / num_worlds
    avg_manhattan = float(manhattan) / num_worlds
    avg_score = float(cumulative_score) / num_worlds
    
    print 'Avg normed: {}'.format(avg_normed)
    print 'Avg manhattan: {}'.format(avg_manhattan)
    print 'Avg score: {}'.format(avg_score)
    results = {'mse': mse, 'quality': avg_normed, 'manhattan': avg_manhattan}
    pickle.dump(results, open(os.path.join(pickle_path, 'results.p'), 'wb'))
    return avg_normed, avg_manhattan 


if __name__=='__main__': 
    run_eval() 