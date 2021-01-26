import os.path

from aws_cdk.aws_s3_assets import Asset

from aws_cdk import (
    core,
    aws_autoscaling as autoscaling,
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_ssm as ssm,
)

from utils import consolidate_context

dirname = os.path.dirname(__file__)

REGION = "us-west-2"
STATUS = "research"

env = core.Environment(account="557762406065", region=REGION)
tags = {
    "CostCentre": "AB1250",
    "Lifespan": "forever",
    "Status": STATUS,
    "ServiceOwner": "FR",
    "ServiceCategory": "Deep Learning",
    "Product": "Deep Learning",
    "Authors": "FR",
    "Project": "NA",
    "Environment": "Dev",
}

class EC2InstanceStack(core.Stack):

    def __init__(self, app, id, target, **kwargs):
        """ Initializer """
        super().__init__(app, id, **kwargs)

        self.config = consolidate_context(self, target)

        # Fetch VPC info
        vpc = ec2.Vpc.from_lookup(self, "VPC", vpc_id=self.config["vpc_id"]["us-west-2"])
        subnets = vpc.private_subnets

        # AMI
        ami = ec2.MachineImage.generic_linux({
                    self.region: self.config["ami_name"]
                })

        # Create a specific SG for DT instances
        ec2_sg = ec2.SecurityGroup(
            self, "EC2SG", description='ec2 SG', vpc=vpc
        )
        # Add default rules for the SG
        self._add_default_rules(ec2_sg, vpc)


        # IAM role and profile for the EC2
        iam_role = iam.Role(
            self,
            "EC2Role",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            inline_policies={
                "extra-permissions": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=["s3:*"],
                            resources=[
                                "arn:aws:iam::aws:policy/AmazonS3FullAccess"
                            ],
                        )
                    ]
                )
            },
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    managed_policy_name="ReadOnlyAccess"
                ),
                # Required by SSM StateManager
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    managed_policy_name="AmazonSSMManagedInstanceCore"
                ),
            ],
        )
        iam_ip = iam.CfnInstanceProfile(
            self, "EC2InstProf", path="/", roles=[iam_role.role_name],
        )

        # Create a placement group in cluster mode to locate all the DT nodes close
        # to each other. This mode means we restrict ourself to a single AZ
        placement_group=ec2.CfnPlacementGroup(
            self, "EC2PG", strategy="cluster"
        )

        instance_specs = self.config["instance_spec"]
        for key, spec in instance_specs.items():

            launch_template = ec2.CfnLaunchTemplate(
                self,
                f"EC2LT{key}",
                launch_template_data=ec2.CfnLaunchTemplate.LaunchTemplateDataProperty(
                    block_device_mappings=[
                        ec2.CfnLaunchTemplate.BlockDeviceMappingProperty(
                            device_name="/dev/xvda",
                            ebs=ec2.CfnLaunchTemplate.EbsProperty(
                                volume_size=self.config["ebs_volume_size"],
                                volume_type="gp2",
                            ),
                        )
                    ],
                    iam_instance_profile=ec2.CfnLaunchTemplate.IamInstanceProfileProperty(
                        arn=iam_ip.attr_arn
                    ),
                    image_id=str(ami.get_image(self).image_id),
                    instance_type=spec["instance_type"],
                    # TODO: we should use SSM Systems Manager rather than native SSH
                    key_name=self.config["key_pair"].format(region=self.region),
                    security_group_ids=[ec2_sg.security_group_id]
                ),
            )

            asg = autoscaling.CfnAutoScalingGroup(
                self,
                f"ASG{key}",
                desired_capacity="1",
                min_size="1",
                max_size="1",
                mixed_instances_policy=autoscaling.CfnAutoScalingGroup.MixedInstancesPolicyProperty(
                    instances_distribution=autoscaling.CfnAutoScalingGroup.InstancesDistributionProperty(
                        on_demand_base_capacity=0,
                        on_demand_percentage_above_base_capacity=spec['on_demand_percentage_above_base_capacity'],
                        spot_allocation_strategy="lowest-price",
                        spot_instance_pools=1,
                    ),
                    launch_template=autoscaling.CfnAutoScalingGroup.LaunchTemplateProperty(
                        launch_template_specification=autoscaling.CfnAutoScalingGroup.LaunchTemplateSpecificationProperty(
                            launch_template_id=launch_template.ref,
                            version=launch_template.attr_latest_version_number,
                        )
                    )
                ),
                # Use placement group, which means we restrict ourself to a single AZ
                placement_group=placement_group.ref,
                # Restrict to a single subnet because of the placement group
                vpc_zone_identifier=[subnets[0].subnet_id],
                # Set max instance lifetime to 7 days for worker nodes and 30 days for master
            )
            # Add a name to the ASG and it will be propagated to underlying EC2 instances
            tag_name = f'{self.config["prefix"]} ASG {key}'
            core.Tags.of(asg).add("Name", tag_name)

    def _add_default_rules(self, sg: ec2.SecurityGroup, vpc: ec2.Vpc):
        """ Adds default rules to given security group.
        TODO: should create a high level construct to hide those rules.
        TODO: need to review those rules because we probably do not need all of them.
        Current default rules are:
            - All IMCP, TCP, and UDP from within the given VPC.
            - All TCP traffic from MetService network.
        """
        # Allow traffic coming from internal MetService network
        # TODO: we should probably restrict this rule to port 22 (SSH)
        sg.add_ingress_rule(
            peer=ec2.Peer.ipv4(self.config["metservice_cidr"]),
            connection=ec2.Port.all_tcp(),
            description="Allow connections from MetService network",
        )
        sg.add_ingress_rule(
            peer=ec2.Peer.ipv4(vpc.vpc_cidr_block),
            connection=ec2.Port.all_tcp(),
            description="Allow all TCP traffic from within the VPC",
        )
        sg.add_ingress_rule(
            peer=ec2.Peer.ipv4(vpc.vpc_cidr_block),
            connection=ec2.Port.all_udp(),
            description="Allow all UDP traffic from within the VPC ",
        )
        sg.add_ingress_rule(
            peer=ec2.Peer.ipv4(vpc.vpc_cidr_block),
            connection=ec2.Port.all_icmp(),
            description="Allow all ICMP traffic from within the VPC ",
        )

app = core.App()
EC2InstanceStack(app,
 "ec2-instance",
 "dev",
 env=env,
 stack_name=f'ResearchDL{app.node.try_get_context("stack_ec2_suffix")}')

app.synth()


