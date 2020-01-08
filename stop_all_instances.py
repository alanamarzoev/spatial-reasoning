import sys
import boto3 


def main(): 
    client = boto3.client('ec2')  

    requests = client.describe_spot_instance_requests()[u'SpotInstanceRequests']
    
    instance_ids = []
    for request in requests: 
        if request[u'State'] == 'cancelled' or request[u'State'] == 'closed':
            continue 
        else: 
            if request[u'LaunchSpecification'][u'KeyName'] == 'alanakey':
                updated_request = request
                request_id = updated_request[u'SpotInstanceRequestId']
                instance_ids.append(request_id)
                # while u'InstanceId' not in updated_request: 
                #     updated_requests = client.describe_spot_instance_requests()[u'SpotInstanceRequests']
                #     for req in updated_requests: 
                #         if req[u'SpotInstanceRequestId'] == request_id: 
                #             updated_request = req 
                                          
                # if u'InstanceId' in updated_request: 
                #     instance_id = updated_request[u'InstanceId']
                #     instance_ids.append(instance_id)

    client.cancel_spot_instance_requests(SpotInstanceRequestIds=instance_ids) 
    print(instance_ids)

if __name__=='__main__': 
    main()