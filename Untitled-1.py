import os 

strdir = "git checkout -f b457b71e8132bb7e368a72063175d7e5f7d9b8bb"
getdata = "./download_data.sh"
os.system(strdir)
os.system(getdata)

# grounded language learning -- baseline experiments script 
# 1. unmodified spatial-reasoning experiments for local and global human annotations 
exp1 = "python2 reinforcement.py --annotations human --mode local --save_path ../exp_results/janner_lstm_embeddings_human_local"
exp2 = "python2 reinforcement.py --annotations human --mode global --save_path ../exp_results/janner_lstm_embeddings_human_global"
os.system(exp1)
os.system(exp2)

chdir = "git checkout -f lstm-embeddings"
os.system(chdir)

# 2. modified spatial-reasoning experiments for local and global human and synthetic annotations 
exp3 = "python2 reinforcement.py --annotations human --mode local --save_path ../exp_results/lstm_embeddings_human_local"
exp4 = "python2 reinforcement.py --annotations human --mode global --save_path ../exp_results/lstm_embeddings_human_global"

exp5 = "python2 reinforcement.py --annotations synthetic --mode local --save_path ../exp_results/lstm_embeddings_synthetic_local --max_train 99 --max_test 10"
exp6 = "python2 reinforcement.py --annotations synthetic --mode global --save_path ../exp_results/lstm_embeddings_synthetic_global --max_train 99 --max_test 10"

os.system(exp3)
os.system(exp4)
os.system(exp5)
os.system(exp6)







