import os.path
from aws_cdk import (App, Environment)

# from constructs import Construct
from ec2_instance_env_setup import EC2InstanceStackSetup
from ec2_instance_stack import EC2InstanceStack

# # Get the instance type
EC2_INSTANCE_TYPE = os.getenv('INSTANCE_TYPE')
if not EC2_INSTANCE_TYPE:
    EC2_INSTANCE_TYPE = 'Simple'

# Choose EC2 instance class based on the type
EC2_INSTANCE_FUNC = None

if EC2_INSTANCE_TYPE == 'Simple':
    EC2_INSTANCE_FUNC = EC2InstanceStack

app = App()
stack_name = "arm-sve-{}".format(EC2InstanceStackSetup.PROJECT_NAME)
sve_stack = EC2_INSTANCE_FUNC(app, stack_name, env=Environment(account="221082202549", region='us-east-2'))
app.synth()
