# Finite Volume Method (FVM) CDK using SHLL and SVE

These cores are ssed as part of research into Scalable Vector Extension (SVE) intrinsic functions executed on AWS Graviton4 processors.

## Target Systems

These codes are designed for execution on AWS instances powered using Graviton4 cores; these codes will not compile and run anywhere else, with the exception of some Nvidia development boards and the Apple M4 processor.

In order to run these, you'll need to make sure that:

* NPM and Nodejs are installed,
* You have configured your python venv correctly, activated it, and have aws-cdk.core and aws-cdk-lib installed,
* You have installed the [AWS CLI tool](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
* You have your AWS credentials properly set up. Test this using:

```bash
aws sts get-caller-identity
```

## Deployment using CDK

To deploy the AWS resources required to run these codes:

* Navigate to the CDK directory;
* Make sure you've installed your node modules:

```bash
npm install
```

* Synthesize your cloud formation template - this is a test to make sure you have properly configured your AWS as well as have the requirements installed:

```bash
npm run cdk synth
```
* Finally, to deploy:

```bash
npm run cdk deploy
```
