cd spatial-reasoning; python3 background.py lstm-local-all-human python2 reinforcement.py --annotations both --mode local --save_path lstm-local-all-human                     --max_train_human 1500 --max_test_human 400 --max_train_synthetic 0 --max_test_synthetic 0                     --epochs 1250 --model full --embedding_type lstm