import os, argparse, pickle, torch
import pipeline, models, data, utils, visualization


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='reinforcement/model.pth')
parser.add_argument('--test_path', type=str, default='reinforcement/test.txt')

args = parser.parse_args()

# load model
model = torch.load(args.model_path)
model.eval()
