import os, math, pickle, torch, numpy as np, pdb, pipeline
from torch.autograd import Variable
import matplotlib; matplotlib.use('Agg')
from matplotlib import cm
from matplotlib import pyplot as plt
from tqdm import tqdm
from pytorch_transformers import *
import environment 
import process_results

from pandas import DataFrame

import sys
import sklearn 
import lsh 
from lsh import LSH
import os
from sklearn.metrics.pairwise import cosine_similarity

from torch.nn import functional as F


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)


def dist_cos(query, target):
    query = query / torch.norm(query, dim=1).unsqueeze(1)
    target = target / torch.norm(target, dim=1).unsqueeze(1)
    return (1 - (query * target).sum(dim=1)) / 2


def get_cos_closest(input_embedding, embeddings_info): 
    closest = []
    num_to_return = 5

    def argmax(pairs):
        return max(pairs, key=lambda x: x[0])

    furthest_within_closest = None
    for line, embedding in embeddings_info.items():
        # dist = np.linalg.norm(input_embedding - embedding)
        dist = dist_cos(input_embedding, embedding) 

        if len(closest) < num_to_return:
            closest.append((dist, line))
        elif dist < furthest_within_closest[0]:
            closest.remove(furthest_within_closest)
            closest.append((dist, line))

        furthest_within_closest = argmax(closest)
        
    return closest 


def get_single_embedding(sentence): 
    words = sentence.strip("\n").split(" ")
    if words[0] == 'ball' or words[0] == 'square' or words[0] == 'triangle': 
        sentence = 'move the {} {} to the {} goal'.format(words[1], words[0], words[2])
    input_ids = torch.tensor([tokenizer.encode(sentence)])  
    out = []
    with torch.no_grad():
        _, _, hiddens = model(input_ids)
        word_rep = hiddens[0].mean(dim=1)
        seq_rep = hiddens[-1].mean(dim=1)
    out.append(F.normalize(word_rep, dim=1))
    out.append(F.normalize(seq_rep, dim=1))
    embedding = torch.cat(out, dim=1)

    return embedding 


def save_synthetic_embeddings(sentences): 
    all_info = {}

    for sentence in sentences: 
        words = sentence.strip("\n").split(" ")
        if words[0] == 'ball' or words[0] == 'square' or words[0] == 'triangle': 
            sentence = 'move the {} {} to the {} goal'.format(words[1], words[0], words[2])
        input_ids = torch.tensor([tokenizer.encode(sentence)])  
        out = []
        with torch.no_grad():
            _, _, hiddens = model(input_ids)
            word_rep = hiddens[0].mean(dim=1)
            seq_rep = hiddens[-1].mean(dim=1)
        out.append(F.normalize(word_rep, dim=1))
        out.append(F.normalize(seq_rep, dim=1))
        embedding = torch.cat(out, dim=1)
        all_info[sentence] = embedding 

    with open('embedded_synthetic.pkl', 'wb') as f: 
        pickle.dump(all_info, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    return all_info


'''
saves predictions of model, targets, and MDP info (rewards / terminal map) as pickle files
inputs is a tuple of (layouts, objects, instruction_indices)
assumes that save path already exists
'''
def save_predictions(model, inputs, targets, rewards, terminal, text_vocab, save_path, max_seq_length, prefix=''):
    ## wrap tensors in Variables to pass to model

    layouts, objects, indices = inputs 
    if prefix == 'test_':
        vocab_text = {}
        for word, ind in text_vocab.items(): 
            vocab_text[ind] = word 
        print('save path : {}'.format(save_path)) 
        no_pickle_path = save_path.replace('pickle', '')
        print('no pickle: {}'.format(no_pickle_path))
	handle = open(os.path.join(no_pickle_path, 'sent_to_ind.pkl'), 'rb')
        sent_to_ind = pickle.load(handle)
        ind_to_sent = {}
        for sent, ind in sent_to_ind.items():
            ind_to_sent[str(ind)] = sent 
        
        sent_to_embedding = save_synthetic_embeddings(sent_to_ind.keys())
        embedding_to_sent = {}
        for sent, embed in sent_to_embedding.items():
            embedding_to_sent[str(embed)] = sent

        data = []
        for sent, embed in sent_to_embedding.items(): 
            data.append(embed.squeeze(0)) 

        data = [x.numpy() for x in data]
        data = np.asarray(data)
    
        lsh_model = LSH(data)
        num_of_random_vectors = 10
        lsh_model.train(num_of_random_vectors)

        similarity_meta_LSH = []
        similarity_meta_COS = []

        new_indices = []
        for index in indices: 
            index = filter(lambda a: a != 0, index.tolist())
            words = [vocab_text[x] for x in index]
            sent = " ".join(words)
            embed = get_single_embedding(sent)
            res = lsh_model.query(embed.squeeze(0).numpy(), 5, 10)
            y = res['id'].tolist() 
            keys = [str(torch.from_numpy(data[x]).unsqueeze(0)) for x in y]  
            results = [embedding_to_sent[key] for key in keys] 
            similarity_meta_LSH.append((sent, results))
            new_ind = sent_to_ind[results[0]]
            new_indices.append(new_ind)
            cos_closest = get_cos_closest(embed, sent_to_embedding)
            similarity_meta_COS.append((sent, cos_closest))
        
        save_path = save_path + "/pickle"
        
        # pickle.dump(similarity_meta_COS, open(os.path.join(save_path, prefix+'cos_closest.p'), 'wb'))
        # pickle.dump(similarity_meta_LSH, open(os.path.join(save_path, prefix+'lsh_closest.p'), 'wb') )
        
        all_new_indices = []
        for ind in new_indices: 
            tensors = []
            x = torch.LongTensor(ind)
            pad_size = max_seq_length - len(x) 
            y = torch.zeros(pad_size, dtype=torch.int64)
            tensors.append(y)
            tensors.append(x)
            tensors = torch.cat(tensors) 
            all_new_indices.append(tensors)

        all_new_indices = torch.stack(all_new_indices).cuda()
        new_inputs = (layouts, objects, all_new_indices)

        all_predictions = []
        x = int(layouts.shape[0] / 99) 
        for i in range(x):
            new_inputs = (layouts[i*99:(i+1)*99], objects[i*99:(i+1)*99], all_new_indices[i*99:(i+1)*99])
            new_input_vars = ( Variable(tensor.contiguous()) for tensor in new_inputs )
            # print('input vars shape: {}'.format(new_input_vars[2].shape))
            predictions_shim = model(new_input_vars)
            predictions_shim = predictions_shim.data.cpu().numpy()
            all_predictions.append(predictions_shim)

        predictions_shim = np.stack(all_predictions)
        predictions_shim = predictions_shim.reshape(-1, *predictions_shim.shape[-3:])
        # predictions_shim = np.reshape(predictions_shim, (predictions_shim.shape[0] * predictions_shim.shape[1], predictions_shim.shape[2], predictions_shim.shape[3]))
        # new_input_vars = ( Variable(tensor.contiguous()) for tensor in new_inputs )

        # import ipdb; ipdb.set_trace() 
	if not os.path.exists(save_path):
		os.makedirs(save_path)
        print('saving to {}'.format(os.path.join(save_path, prefix+'predictions_shim.p')))
        pickle.dump(predictions_shim, open(os.path.join(save_path, prefix+'predictions_shim.p'), 'wb') )
    #else: 
    #    save_path = save_path + "/pickle"


    # input_vars = ( Variable(tensor.contiguous()) for tensor in inputs )
    # predictions = model(input_vars)

    all_predictions = []
    x = int(layouts.shape[0] / 99) 
    for i in range(x):
        new_inputs = (layouts[i*99:(i+1)*99], objects[i*99:(i+1)*99], indices[i*99:(i+1)*99])
        new_input_vars = ( Variable(tensor.contiguous()) for tensor in new_inputs )
        # print('input vars shape: {}'.format(new_input_vars[2].shape))
        predictions = model(new_input_vars)
        predictions = predictions.data.cpu().numpy()
        all_predictions.append(predictions)

    predictions = np.stack(all_predictions)
    predictions = predictions.reshape(-1, *predictions.shape[-3:])
    # predictions = np.reshape(predictions, (predictions.shape[0] * predictions.shape[1], predictions.shape[2], predictions.shape[3]))
    ## convert to numpy arrays for saving to disk
    # predictions = predictions.data.cpu().numpy()
    targets = targets.cpu().numpy()[:len(predictions)]
    ## save the predicted and target value maps
    ## as well as info about the MDP and instruction
    pickle.dump(inputs, open(os.path.join(save_path, prefix+'inputs.p'), 'wb') )
    pickle.dump(predictions, open(os.path.join(save_path, prefix+'predictions.p'), 'wb') )
    pickle.dump(targets, open(os.path.join(save_path, prefix+'targets.p'), 'wb') )
    pickle.dump(rewards, open(os.path.join(save_path, prefix+'rewards.p'), 'wb') )
    pickle.dump(terminal, open(os.path.join(save_path, prefix+'terminal.p'), 'wb') )
    pickle.dump(text_vocab, open(os.path.join(save_path, prefix+'vocab.p'), 'wb') )

    if prefix == 'test_':
        print("REGULAR:")
        reg_res = process_results.run_eval(save_path, prefix=prefix, shim=False)
        print("SHIM: ")
        shim_res = process_results.run_eval(save_path, prefix=prefix, shim=True)

        pickle.dump(reg_res, open(os.path.join(save_path, prefix+'reg_res.p'), 'wb') )
        pickle.dump(shim_res, open(os.path.join(save_path, prefix+'shim_res.p'), 'wb') )


'''
test set is dict from
world number --> (state_obs, goal_obs, instruct_inds, values)
'''

# def evaluate(model, test_set, savepath=None):
#     progress = tqdm(total=len(test_set))
#     count = 0
#     for key, (state_obs, goal_obs, instruct_words, instruct_inds, targets) in test_set.iteritems():
#         progress.update(1)
        
#         state = Variable( torch.Tensor(state_obs).long().cuda() )
#         objects = Variable( torch.Tensor(goal_obs).long().cuda() )
#         instructions = Variable( torch.Tensor(instruct_inds).long().cuda() )
#         targets = torch.Tensor(targets)
#         # print state.size(), objects.size(), instructions.size(), targets.size()
        
#         preds = model.forward( (state, objects, instructions) ).data.cpu()

#         state_dim = 1
#         for dim in state.size()[-2:]:
#             state_dim *= dim

#         if savepath:
#             num_goals = preds.size(0) / state_dim
#             for goal_num in range(num_goals):
#                 lower = goal_num * state_dim
#                 upper = (goal_num + 1) * state_dim
#                 fullpath = os.path.join(savepath, \
#                             str(key) + '_' + str(goal_num) + '.png')
#                 pred = preds[lower:upper].numpy()
#                 targ = targets[lower:upper].numpy()
#                 instr = instruct_words[lower]

#                 pipeline.visualize_value_map(pred, targ, fullpath, title=instr)


# def get_children(M, N):
#     children = {}
#     for i in range(M):
#         for j in range(N):
#             pos = (i,j)
#             children[pos] = []
#             for di in range( max(i-1, 0), min(i+1, M-1)+1 ):
#                 for dj in range( max(j-1, 0), min(j+1, N-1)+1 ):
#                     child = (di, dj)
#                     if pos != child and (i == di or j == dj):
#                         children[pos].append( child )
#     return children


# '''
# values is M x N map of predicted values
# '''
# def get_policy(values):
#     values = values.squeeze()
#     M, N = values.shape
#     states = [(i,j) for i in range(M) for j in range(N)]
#     children = get_children( M, N )
#     policy = {}
#     for state in states:
#         reachable = children[state]
#         selected = sorted(reachable, key = lambda x: values[x], reverse=True)
#         policy[state] = selected[0]
#     return policy

# def simulate(model, sim_set):
#     # progress = tqdm(total=len(test_set))
#     steps_list = []
#     count = 0
#     for key in tqdm(range(len(sim_set))):
#         (state_obs, goal_obs, instruct_words, instruct_inds, targets, mdps) = sim_set[key]
#         # progress.update(1)
#         # print torch.Tensor(state_obs).long().cuda()
#         state = Variable( torch.Tensor(state_obs).long().cuda() )
#         objects = Variable( torch.Tensor(goal_obs).long().cuda() )
#         instructions = Variable( torch.Tensor(instruct_inds).long().cuda() )
#         targets = torch.Tensor(targets)
#         # print state.size(), objects.size(), instructions.size()
        
#         preds = model.forward(state, objects, instructions).data.cpu().numpy()
#         # print 'sim preds: ', preds.shape

#         ## average over all goals
#         num_goals = preds.shape[0]
#         for ind in range(num_goals):
#             # print ind
#             mdp = mdps[ind]
#             values = preds[ind,:]
#             dim = int(math.sqrt(values.size))
#             positions = [(i,j) for i in range(dim) for j in range(dim)]
#             # print 'dim: ', dim
#             values = preds[ind,:].reshape(dim, dim)
#             policy = mdp.get_policy(values)

#             # plt.clf()
#             # plt.pcolor(policy)


#             ## average over all start positions
#             for start_pos in positions:
#                 steps = mdp.simulate(policy, start_pos)
#                 steps_list.append(steps)
#                 # pdb.set_trace()
#                 # print 'simulating: ', start_pos, steps
#     avg_steps = np.mean(steps_list)
#     # print 'avg steps: ', avg_steps, len(steps_list), len(sim_set), num_goals
#     return avg_steps








