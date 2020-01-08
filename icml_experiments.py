import os
import sys 
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
    spot_price = '0.5'

    # model_setup = {'lstm': 'full', 'bert-word': 'bert-full', 'bert-word-fixed': 'bert-full'}
    model_setup = {'lstm': 'full'}
    data_splits = {'all-human': {'max_train_human': 1500, 'max_train_synthetic': 0, 'max_test_human': 400, 'max_test_synthetic': 0}, 
                'all-synthetic':  {'max_train_human': 0, 'max_train_synthetic': 1500, 'max_test_human': 400, 'max_test_synthetic': 0}, 
                'human-and-synthetic':  {'max_train_human': 1500, 'max_train_synthetic': 1500, 'max_test_human': 400, 'max_test_synthetic': 0}}
    modes = ['local', 'global']

    commands = []
    variant_to_cmd = {}
    cmd_to_variant = {}
    for mode in modes: 
        for embed, model in model_setup.items(): 
            for name, split in data_splits.items(): 
                save_path = '{}-{}-{}'.format(embed, mode, name)
                cmd = 'cd spatial-reasoning; python3 background.py {} python2 reinforcement.py --annotations both --mode {} --save_path {} \
                    --max_train_human {} --max_test_human {} --max_train_synthetic {} --max_test_synthetic {} \
                    --epochs {} --model {} --embedding_type {}'.format(save_path, 
                                                                    mode, 
                                                                    save_path, 
                                                                    split['max_train_human'],
                                                                    split['max_test_human'],
                                                                    split['max_train_synthetic'], 
                                                                    split['max_test_synthetic'], 
                                                                    1250, 
                                                                    model, 
                                                                    embed)
                commands.append(cmd)
                variant_to_cmd[save_path] = cmd 
                cmd_to_variant[cmd] = save_path 

    instance_count = len(modes) * len(model_setup.items()) * len(data_splits.items())
    instance_count = 1
    client = boto3.client('ec2')
    ec2 = boto3.resource('ec2')

    instance_args = dict(
        ImageId=image_id,
        KeyName=keypair_name,
        InstanceType=instance_type,
        EbsOptimized=False,
        BlockDeviceMappings=[
            dict(DeviceName="/dev/xvdb", Ebs=dict(VolumeSize=100)),
            ]
    )
    print("instance count: {}".format(instance_count))
    spot_args = dict(
        DryRun=False,
        InstanceCount=instance_count,
        LaunchSpecification=instance_args,
        SpotPrice=spot_price,
    )

    import pprint 

    print('spot args: {}'.format(spot_args))
    # response = client.request_spot_instances(**spot_args)
    # print('Launched EC2 job - Server response:')
    # pprint.pprint(response)
    # print('*****'*5)
    # spot_request_id = response['SpotInstanceRequests'][0]['SpotInstanceRequestId']
    pending = []
    instance_ids = []
    time.sleep(30)
    requests = client.describe_spot_instance_requests()[u'SpotInstanceRequests']
    for request in requests: 
        if request[u'State'] == 'cancelled' or request[u'State'] == 'closed':
            continue 
        else: 
            if request[u'LaunchSpecification'][u'KeyName'] == 'alanakey' and request[u'LaunchSpecification'][u'InstanceType'] == instance_type:
                pprint.pprint(request)
                updated_request = request
                request_id = updated_request[u'SpotInstanceRequestId'] 
                while u'InstanceId' not in updated_request: 
                    updated_requests = client.describe_spot_instance_requests()[u'SpotInstanceRequests']
                    for req in updated_requests: 
                        if req[u'SpotInstanceRequestId'] == request_id: 
                            updated_request = req 
                                          
                if u'InstanceId' in updated_request: 
                    instance_id = updated_request[u'InstanceId']
                    instance_ids.append(instance_id)
    
    print(instance_ids)

    response = client.describe_instances(
        InstanceIds=instance_ids,
        DryRun=False,
    )[u'Reservations']

    ip_addrs = []
    for res in response: 
        instances = res[u'Instances']
        for i in instances: 
            # import ipdb; ipdb.set_trace()
            ip = i[u'PublicIpAddress']
            ip_addrs.append(ip)
    
    print('ip addrs: {}'.format(ip_addrs))

    tracker = {}
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    for i, ip in enumerate(ip_addrs): 
        string = 'ahhhhhhhhhh'
        out = None
        while out != string:
            time.sleep(2)
            print('out: {}'.format(out))
            cmd = 'ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -o ConnectionAttempts=1 -i ~/Downloads/alanakey.pem.txt ubuntu@{} echo {}'.format(ip, string)
            print(cmd)
            p = subprocess.Popen([
                'ssh', 
                '-o', 'StrictHostKeyChecking=no',
                '-o', 'ConnectTimeout=10',
                '-o', 'ConnectionAttempts=1',
                '-i', '~/Downloads/alanakey.pem.txt',
                'ubuntu@{}'.format(ip),
                'echo {}'.format(string)
            ], universal_newlines=True, stdin=open('/dev/null', 'w'), stdout=subprocess.PIPE, stderr=open('/dev/null', 'w'))
            (out, _) = p.communicate()
            out = out.strip()
        
        os.system('rsync -avze "ssh -o StrictHostKeyChecking=no -i ~/Downloads/{}.pem.txt" --progress --del ~/spatial-reasoning ubuntu@{}:'.format(keypair_name, ip)) 
        cmds = ['python2 -m pip install --user torch', 'python2 -m pip install --user tqdm', 'python2 -m pip install --user torchvision', 'python2 -m pip install --user pytorch-transformers']
        
        for cmd in cmds: 
            os.system('ssh -o StrictHostKeyChecking=no -i ~/Downloads/{}.pem.txt ubuntu@{} "{}"'.format(keypair_name, ip, cmd))
        
        os.system('echo "{}" > test.sh'.format(commands[i]))
        os.system('rsync -avze "ssh -o StrictHostKeyChecking=no -i ~/Downloads/{}.pem.txt" --progress --del ~/spatial-reasoning/test.sh ubuntu@{}:'.format(keypair_name, ip)) 
        os.system('ssh -o StrictHostKeyChecking=no -i ~/Downloads/{}.pem.txt ubuntu@{} chmod +x test.sh'.format(keypair_name, ip))
        os.system('ssh -o StrictHostKeyChecking=no -i ~/Downloads/{}.pem.txt ubuntu@{} sudo ./test.sh'.format(keypair_name, ip))
        
        variant = cmd_to_variant[commands[i]]
        tracker[variant] = {'cmd': commands[i], 'ip_addr': ip}

    print(tracker)

if __name__ == '__main__':
    main()