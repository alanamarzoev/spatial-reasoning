import os
import sys 
import logging
import boto3
import time
import datetime
import subprocess

from collections import defaultdict
from itertools import product
from botocore.exceptions import ClientError


def create_ec2_instance(image_id, instance_type, keypair_name):
    # Provision and launch the EC2 instance
    ec2_client = boto3.client('ec2')
    try:
        response = ec2_client.run_instances(ImageId=image_id,
                                            InstanceType=instance_type,
                                            KeyName=keypair_name,
                                            BlockDeviceMappings=[
                                                dict(DeviceName="/dev/xvdb", Ebs=dict(VolumeSize=100))], 
                                            MinCount=1,
                                            MaxCount=1)
    except ClientError as e:
        logging.error(e)
        return None
    return response['Instances'][0]


def main(): 
    image_id = 'ami-015cba4cac78bfd3d'
    instance_type = 'p2.xlarge'
    keypair_name = 'alanakey'

    model_setup = {'lstm': 'full', 'bert-word': 'bert-full', 'bert-word-fixed': 'bert-full'}
    data_splits = {'all-human': {'max_train_human': 1500, 'max_train_synthetic': 0, 'max_test_human': 400, 'max_test_synthetic': 0}, 
                'all-synthetic':  {'max_train_human': 0, 'max_train_synthetic': 1500, 'max_test_human': 400, 'max_test_synthetic': 0}, 
                'human-and-synthetic':  {'max_train_human': 1500, 'max_train_synthetic': 1500, 'max_test_human': 400, 'max_test_synthetic': 0}}
    modes = ['local', 'global']

    for mode in modes: 
        for embed, model in model_setup.items(): 
            for split in data_splits.values(): 
                save_path = ''
                cmd = 'python2 reinforcement.py --annotations both --mode {} --save_path {} \
                    --max_train_human {} --max_test_human {} --max_train_synthetic {} --max_test_synthetic {} \
                    --epochs {} --model {} --embedding_type {}'.format(mode, 
                                                                    save_path, 
                                                                    split['max_train_human'],
                                                                    split['max_test_human'],
                                                                    split['max_train_synthetic'], 
                                                                    split['max_test_synthetic'], 
                                                                    1250, 
                                                                    model, 
                                                                    embed)
            








def main():
    params = {
        'seed': [333],
        'no_vis': [False],
        'render': [True],
        'cuda': [True],
        'log_interval': [1],
        'loss_type': ['kl'],
        'n_envs': [3000],
        'max_corrections': [6],
        'traj_subsample': [25],
        'embedding_dim': [16],
        'obs_net':      [dict(filter_sizes=((8, 2), (8, 2)), hidden_sizes=(32,), flat_dim=200, output_dim=32,
                        batchnorm=True)],
        'traj_obs_net': [dict(filter_sizes=((4, 2), (4, 2)), hidden_sizes=(16, ), flat_dim=100, output_dim=16,
                        batchnorm=True)],
        'prev_traj_encoder':  [dict(hidden_sizes=(16,), output_dim=4, filter_sizes=((8, 2),), flat_size=8, pool=True)],
        'correction_encoder': [dict(hidden_sizes=(16,), output_dim=256, filter_sizes=((8, 2),), flat_size=16, )],
        'hlg_encoder':        [dict(hidden_sizes=(16,), output_dim=256, filter_sizes=((8, 2),), flat_size=24, )],
        'correction_traj_encoder': [dict(hidden_sizes=(32, ), output_dim=32)],
        'extra_obs_net':           [dict(hidden_sizes=(16, ), output_dim=16)],
        'prob_network':            [dict(hidden_sizes=(64, 64),batchnorm=True)],

        'ablation': [None],
        'method': ['lgpl']
    }

    def get_exps(inp):
        return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))
    
    variants = list(get_exps(params))

    num_instances = len(variants)

    

    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s: %(asctime)s: %(message)s')

    # for i in range(num_instances): 
    #     instance_info = create_ec2_instance(image_id, instance_type, keypair_name)
    #     if instance_info is not None:
    #         logging.info(f'Launched EC2 Instance {instance_info["InstanceId"]}')
    #         logging.info(f'    VPC ID: {instance_info["VpcId"]}')
    #         logging.info(f'    Private IP Address: {instance_info["PrivateIpAddress"]}')
    #         logging.info(f'    Current State: {instance_info["State"]["Name"]}')
    
    # time.sleep(20)

    ec2 = boto3.resource('ec2')

    running_instances = ec2.instances.filter(Filters=[{
        'Name': 'instance-state-name',
        'Values': ['running']}])

    ec2info = defaultdict()
    ip_addrs = []
    for instance in running_instances:  
        ip_addrs.append(instance.public_ip_address) 

    
    print("IP ADDRS: {}".format(ip_addrs))
    tracker = {}
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    for i, ip in enumerate(ip_addrs): 
        string = 'ahhhhhhhhhh'
        out = None
        while out != string:
            time.sleep(2)
            p = subprocess.Popen([
                'ssh', 
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'ConnectTimeout=10',
                '-o', 'ConnectionAttempts=1',
                '-i', 'gondor-1.pem',
                'ubuntu@{}'.format(ip),
                'echo {}'.format(string)
            ], universal_newlines=True, stdin=open('/dev/null', 'w'), stdout=subprocess.PIPE, stderr=open('/dev/null', 'w'))
            (out, _) = p.communicate()
            out = out.strip()

        os.system('rsync -avze "ssh -o StrictHostKeyChecking=no -i gondor-1.pem" --progress --del ~/irl ubuntu@{}:'.format(ip)) 
        if False:
            # ip = "18.216.231.124"
            # os.system('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo mkdir -p /home/ubuntu/irl'.format(ip))
            # os.system('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo chmod 777 /home/ubuntu'.format(ip))    
            os.system('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo mkdir -p /mnt/xvdb'.format(ip))
            os.system('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo mkfs /dev/xvdb'.format(ip))
            os.system('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo mount /dev/xvdb /mnt/xvdb'.format(ip))
            os.system('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo apt update -y'.format(ip))
            os.system('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo apt install -y docker'.format(ip))
            os.system('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo service docker start'.format(ip))
            os.system('''ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} 'sudo bash -c "echo '"'"'{{\\"data-root\\": \\"/mnt/xvdb\\"}}'"'"' > /etc/docker/daemon.json"' '''.format(ip))
            os.system('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo systemctl daemon-reload'.format(ip))
            os.system('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo systemctl restart docker'.format(ip))
            os.system('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo docker pull alanamarzoev/irl'.format(ip))
            
        print("RUNNING VARIANT: {}".format(variants[i]))
        variant_hash = str(hash(str(variants[i]))) + str(st)
        variant_hash = "".join(x for x in variant_hash if x.isalnum())
        variant_hash = "original"
        tracker[variant_hash] = variants[i]
        print('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo docker run -v /home/ubuntu/irl:/root/code/irl alanamarzoev/irl:latest bash -c \\\'python /root/code/irl/exps/minigrid/train_lgpl.py --exp_dir {} --variant_num {}\\\''.format(ip, variant_hash, i))
        os.system('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo docker run -v /home/ubuntu/irl:/root/code/irl alanamarzoev/irl:latest bash -c \\\'python /root/code/irl/exps/minigrid/train_lgpl.py --exp_dir {} --variant_num {}\\\''.format(ip, variant_hash, i))
        # os.system('ssh -o StrictHostKeyChecking=no -i gondor-1.pem ubuntu@{} sudo docker run -v /home/ubuntu/irl:/root/code/irl alanamarzoev/irl:latest python irl/upload_results.py --exp_dir {}'.format(ip, variant_hash))
        break 

    with open('{}-experiment-metadata.txt'.format(st), 'a+') as f: 
        f.write(str(tracker))

if __name__ == '__main__':
    main()