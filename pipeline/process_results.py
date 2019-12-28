import os, torch, sys, subprocess, pickle
import matplotlib.pyplot as plt
import numpy as np 
import data
import pipeline
import environment 
import argparse
from score_iteration import ScoreIteration


def get_states(M, N):
    states = [(i,j) for i in range(M) for j in range(N)]
    return states

def get_children(M, N):
    children = {}
    for i in range(M):
        for j in range(N):
            pos = (i,j)
            children[pos] = []
            for di in range( max(i-1, 0), min(i+1, M-1)+1 ):
                for dj in range( max(j-1, 0), min(j+1, N-1)+1 ):
                    child = (di, dj)
                    if pos != child and (i == di or j == dj):
                        children[pos].append( child )
    return children

STATES = get_states(10,10)
CHILDREN = get_children(10,10)
GAMMA = 0.95

def get_policy(values):
    values = values.squeeze()
    policy = {}
    for state in STATES:
        reachable = CHILDREN[state]
        selected = sorted(reachable, key = lambda x: values[x], reverse = True)
        policy[state] = selected
    return policy



def simulate_single(reward_map, terminal_map, approx_values, start_pos, max_steps = 75):
    M, N = reward_map.shape
    if M != 10 or N != 10:
        raise RuntimeError( 'wrong size: {}x{}, expected: {}x{}'.format(M, N, self.M, self.N) )
    policy = get_policy(approx_values)

    pos = start_pos
    visited = set([pos])
    trajectory = []

    total_reward = 0
    for step in range(max_steps):
        val = approx_values[pos]
        rew = reward_map[pos]
        term = terminal_map[pos]
        trajectory.append( (pos, val, rew, term) )

        total_reward += rew * (GAMMA ** step)
        if term:
            break

        reachable = policy[pos]
        selected = 0
        while selected < len(reachable) and reachable[selected] in visited:
            selected += 1
        if selected == len(reachable):
            selected = 0
            break

        pos = reachable[selected]
        visited.add(pos)
    return total_reward



def run_eval(pickle_path, prefix='', shim=False):  
    metric_path = pickle_path 
    quality_path = os.path.join( metric_path, 'quality')
    if not os.path.exists( quality_path ):
        subprocess.call(['mkdir', quality_path])

    save_path = pickle_path 
    
    if shim: 
        predictions = pickle.load( open(os.path.join(save_path, prefix+'predictions_shim.p'), 'rb') ).squeeze()
    else:
        predictions = pickle.load( open(os.path.join(save_path, prefix+'predictions.p'), 'rb') ).squeeze()
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
        si = ScoreIteration(mdp, pred)
        avg_pred, scores_pred = si.iterate()

        mdp = environment.MDP(None, rew, term)
        si = ScoreIteration(mdp, targ)
        avg_targ, scores_targ = si.iterate()

        mdp = environment.MDP(None, rew, term)
        si = ScoreIteration(mdp, unif)
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
    pickle_path = 'logs/trial/synthetic/pickle'
    run_eval(pickle_path) 
