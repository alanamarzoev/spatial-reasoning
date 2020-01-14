#!/om/user/janner/anaconda2/envs/pytorch/bin/python

import os, argparse, pickle, torch
import pipeline, models, data, utils, visualization

parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='logs/trial') 
parser.add_argument('--max_train', type=int, default=5000)
parser.add_argument('--max_test', type=int, default=500)

parser.add_argument('--mode', type=str, default='local', choices=['local', 'global'])
parser.add_argument('--annotations', type=str, default='human', choices=['synthetic', 'human', 'both'])
parser.add_argument('--model', type=str, default='full', choices=['full', 'bert-full', 'no-gradient', 'cnn-lstm', 'uvfa-text'])

parser.add_argument('--map_dim', type=int, default=10)
parser.add_argument('--state_embed', type=int, default=1)
parser.add_argument('--obj_embed', type=int, default=7)

parser.add_argument('--lstm_inp', type=int, default=15)
parser.add_argument('--lstm_hid', type=int, default=30)
parser.add_argument('--lstm_layers', type=int, default=1)
parser.add_argument('--attention_kernel', type=int, default=3)
parser.add_argument('--attention_out_dim', type=int, default=1)

parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--replay_size', type=int, default=100000)
parser.add_argument('--learn_start', type=int, default=1000)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=1250)

parser.add_argument('--max_train_human', type=int, default=1250)
parser.add_argument('--max_test_human', type=int, default=1250)
parser.add_argument('--max_train_synthetic', type=int, default=1250)
parser.add_argument('--max_test_synthetic', type=int, default=1250)

parser.add_argument('--embedding_type', type=str, default='lstm', choices=['lstm', 'bert', 'one-hot', 'random', 'bert-fixed', 'bert-word-fixed', 'bert-word', 'gpt'])
parser.add_argument('--bert_reshape_type', type=str, default='linear', choices=['linear', 'MLP'])

args = parser.parse_args()

#################################
############## Data #############
#################################

train_data, test_data = data.load(args.mode, args.annotations, args.max_train_human, args.max_test_human, args.max_train_synthetic, args.max_test_synthetic)
layout_vocab_size, object_vocab_size, text_vocab_size, text_vocab = data.get_statistics(train_data, test_data)
 
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
pickle.dump(text_vocab, open(os.path.join(args.save_path, 'vocab.p'), 'wb') )

print '\n<Main> Converting to tensors'
train_layouts, train_objects, train_rewards, train_terminal, \
        train_instructions, train_indices, train_values, train_goals, max_len = data.to_tensor_lstm(train_data, text_vocab, args.save_path, train=True)

test_layouts, test_objects, test_rewards, test_terminal, \
    test_instructions, test_indices, test_values, test_goals, max_len = data.to_tensor_lstm(test_data, text_vocab, args.save_path, train=False)

print '<Main> Training:', train_layouts.size(), 'x', train_objects.size(), 'x', train_indices.size()
print '<Main> Rewards: ', train_rewards.size(), '    Terminal: ', train_terminal.size()
print '<Main> Test    :', test_layouts.size(), 'x', test_objects.size(), 'x', test_indices.size()
print '<Main> Rewards: ', test_rewards.size(), '    Terminal: ', test_terminal.size()


#################################
############ Training ###########
#################################

print '\n<Main> Initializing model: {}'.format(args.model)
model = models.init(args, layout_vocab_size, object_vocab_size, text_vocab_size)
target_model = models.init(args, layout_vocab_size, object_vocab_size, text_vocab_size)

## initialize agent
agent = pipeline.Agent(model,  target_model, map_dim = args.map_dim, instr_len = train_indices.size(1), 
                                batch_size = args.batch_size, learn_start = args.learn_start, 
                                replay_size = args.replay_size, lr = args.lr, gamma = args.gamma)

train_inputs = (train_layouts, train_objects, train_indices)
test_inputs = (test_layouts, test_objects, test_indices)

## train agent
scores = agent.train( train_inputs, train_rewards, train_terminal,
                      test_inputs, test_rewards, test_terminal, epochs = args.epochs, save_path = args.save_path)


#################################
######## Save predictions #######
#################################

## make logging directories
pickle_path = os.path.join(args.save_path, 'pickle')
#utils.mkdir(args.save_path)
#utils.mkdir(pickle_path)
if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

if not os.path.exists(pickle_path):
    os.makedirs(pickle_path)

print '\n<Main> Saving model and scores to {}'.format(args.save_path)
## save model
torch.save(model, os.path.join(args.save_path, 'model.pth'))
torch.save(model.state_dict(), os.path.join(args.save_path, 'model_statedict.pth'))

## save scores from training
score_path = os.path.join(pickle_path, 'scores.p')
pickle.dump(scores, open(score_path, 'wb') )


print '<Main> Saving predictions to {}'.format(pickle_path)
## save inputs, outputs, and MDP info (rewards and terminal maps)
pipeline.save_predictions(model, train_inputs, train_values, train_rewards, train_terminal, text_vocab, pickle_path, max_len, prefix='train_')
pipeline.save_predictions(model, test_inputs, test_values, test_rewards, test_terminal, text_vocab, pickle_path, max_len, prefix='test_')






