import os.path
import abc
from aws_cdk.aws_s3_assets import Asset

from aws_cdk import (
    aws_ec2 as ec2,
    aws_iam as iam,
    Stack
)

from constructs import Construct


class EC2InstanceStackSetup(Stack):

    dirname = os.path.dirname(__file__)

    # Manage deployment
    DEPLOYMENT_ENVIRONMENT = 'dev'

    # Configure a project; use test by default
    PROJECT_NAME = os.getenv('PROJECT_NAME')
    if not PROJECT_NAME:
        PROJECT_NAME = 'matt'

    # Configure a region; use eu-west-1 by default
    AWS_REGION = os.getenv('AWS_REGION')
    if not AWS_REGION:
        AWS_REGION = 'us-east-2'

    # Get the instance size ('m8g.medium' = 1x vcpu, 'm8g.large' = 2x vcpu)
    EC2_INSTANCE = os.getenv('EC2_INSTANCE')
    if not EC2_INSTANCE:
        EC2_INSTANCE = 'm8g.large'

    # control string
    CONTROL_STRING = 'https://eu-west-1.console.aws.amazon.com/systems-manager/session-manager/{}?region=eu-west-1'

    # TODO: we might need to come back and change these: The MPI cluster probably won't need the /dev/sdb volume.
    # resources
    RESOURCES_NAME = "BlockDeviceMappings"
    RESOURCES_VALUE = [{
        "DeviceName": "/dev/xvda",
        "Ebs": {
            "VolumeSize": "32",
            "VolumeType": "gp3",
            "DeleteOnTermination": "true"
        }
    }]

    print("==== SUMMARY OF PARAMETERS FOR DEPLOYMENT ====")
    print("DEPLOYMENT_ENVIRONMENT: ", DEPLOYMENT_ENVIRONMENT)
    print("REGION: ", AWS_REGION)
    print("PROJECT: ", PROJECT_NAME)

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # New VPC
        vpc = ec2.Vpc(self, "Vpc", vpc_name="mantel-sve-vpc", max_azs=1)
        # Existing vpc from lookup
        #vpc = ec2.Vpc.from_lookup(self, "VPC", vpc_name="mantel-sve-vpc", is_default=False, region='us-east-2')

        # Use amazon linux 2
        amzn_linux = ec2.MachineImage.latest_amazon_linux2023(cpu_type=ec2.AmazonLinuxCpuType.ARM_64)
        # amzn_linux = ec2.MachineImage.latest_amazon_linux2(cpu_type=ec2.AmazonLinuxCpuType.ARM_64)

        # Instance Role and SSM Managed Policy
        role = iam.Role(self, "InstanceSSM",
                        assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"))

        role.add_managed_policy(iam.ManagedPolicy.from_aws_managed_policy_name(
            "AmazonSSMManagedInstanceCore"))

        self.vpc = vpc
        self.role = role
        self.ami = amzn_linux        

    @abc.abstractmethod
    def define_ec2_instance():
        pass

    def define_assert(self, script_name):
        # Script in S3 as Asset
        asset = Asset(self, "Asset", path=os.path.join(
            self.dirname, script_name))
        asset.grant_read(self.role)
        return asset
