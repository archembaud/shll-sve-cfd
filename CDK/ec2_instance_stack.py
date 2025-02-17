
from aws_cdk import (
    aws_ec2 as ec2,
    CfnOutput
)

from constructs import Construct
from ec2_instance_env_setup import EC2InstanceStackSetup

SCRIPT_NAME = "configure.sh"


class EC2InstanceStack(EC2InstanceStackSetup):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        asset = self.define_assert(SCRIPT_NAME)

        # create single Instance
        self.define_ec2_instance(
            1,  self.vpc, self.ami, self.role, asset)

    def define_assert(self, script_name):
        asset = super().define_assert(script_name)
        return asset

    def define_ec2_instance(self, instance_index, vpc, amzn_linux, role, asset):

        instance = ec2.Instance(self, f"Instance{instance_index}",
                                instance_type=ec2.InstanceType(
                                    self.EC2_INSTANCE),
                                machine_image=amzn_linux,
                                vpc=vpc,
                                role=role)

        # Override resources
        instance.instance.add_property_override(
            self.RESOURCES_NAME, self.RESOURCES_VALUE)

        # # Script in S3 as Asset
        local_path = instance.user_data.add_s3_download_command(
            bucket=asset.bucket,
            bucket_key=asset.s3_object_key
        )

        # # Userdata executes script from S3
        instance.user_data.add_execute_file_command(
            file_path=local_path
        )

        # Export the value of the instance id
        instance_id = instance.instance_id
        control_string = self.CONTROL_STRING.format(instance_id)
        CfnOutput(self, f"InstanceID{instance_index}",
                  value=instance.instance_id)
        CfnOutput(
            self, f"Console Access{instance_index}", value=control_string)
