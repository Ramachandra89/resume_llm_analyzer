# AWS SageMaker Deployment Guide

## Overview
This guide explains how to deploy the Resume Coach application with a SageMaker-hosted LLM endpoint.

## Architecture
- **Frontend**: Streamlit app running on a local machine or EC2 instance
- **Backend API**: FastAPI server on a local machine or EC2 instance
- **LLM**: Meta-Llama-3.1-8B-Instruct hosted on AWS SageMaker

## Prerequisites
1. AWS Account with appropriate permissions (SageMaker, EC2, IAM)
2. Python 3.9+
3. AWS CLI configured with credentials
4. Familiarity with SageMaker and AWS concepts

## Step 1: Set Up Model Hosting

This project supports either:
- a SageMaker endpoint
- a local EC2 model deployment using S3 or Hugging Face artifacts

### 1.1 Create an IAM Role for SageMaker
```bash
aws iam create-role \
  --role-name sagemaker-resume-coach \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Principal": {
          "Service": "sagemaker.amazonaws.com"
        },
        "Action": "sts:AssumeRole"
      }
    ]
  }'

aws iam attach-role-policy \
  --role-name sagemaker-resume-coach \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

### 1.2 Deploy the LLM Endpoint
You can deploy using the SageMaker Console or via boto3. Here's a Python script example:

```python
import boto3
import json
#s3://amazon-sagemaker-547741151022-us-east-1-ebc412e56193
sagemaker_client = boto3.client('sagemaker', region_name='us-east-1')

# Deploy Meta-Llama-3.1-8B-Instruct model
model_name = 'meta-llama-3-1-8b-instruct'
initial_instance_count = 1
instance_type = 'ml.g4dn.xlarge'  # GPU instance (adjust based on your needs)

# Create the endpoint configuration
endpoint_config_name = f'{model_name}-config'
sagemaker_client.create_model(
    ModelName=model_name,
    PrimaryContainer={
        'Image': '763104330519.dkr.ecr.us-east-1.amazonaws.com/huggingface-text-generation:latest',
        'ModelDataUrl': 's3://sagemaker-us-east-1-ACCOUNT_ID/model.tar.gz',
        'Environment': {
            'HF_MODEL_ID': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'HF_TASK': 'text-generation',
            'SM_NUM_GPUS': '1'
        }
    },
    ExecutionRoleArn='arn:aws:iam::ACCOUNT_ID:role/sagemaker-resume-coach'
)

# Create endpoint configuration
sagemaker_client.create_endpoint_config(
    EndpointConfigName=endpoint_config_name,
    ProductionVariants=[
        {
            'VariantName': 'primary',
            'ModelName': model_name,
            'InitialInstanceCount': initial_instance_count,
            'InstanceType': instance_type
        }
    ]
)

# Create the endpoint
endpoint_name = 'resume-coach-endpoint'
sagemaker_client.create_endpoint(
    EndpointName=endpoint_name,
    EndpointConfigName=endpoint_config_name
)

print(f"Endpoint {endpoint_name} creation started. Check SageMaker console for status.")
```

### 1.3 Wait for Endpoint to be In Service
Check status via AWS Console or CLI:
```bash
aws sagemaker describe-endpoint \
  --endpoint-name resume-coach-endpoint \
  --region us-east-1 \
  --query 'EndpointStatus'
```

## Step 2: Configure Environment Variables

Create a `.env` file in the project root:
```env
SAGEMAKER_ENDPOINT_NAME=resume-coach-endpoint
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
BACKEND_API_URL=http://localhost:8000
```

Or export as environment variables:
```bash
export SAGEMAKER_ENDPOINT_NAME=resume-coach-endpoint
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=<your-access-key>
export AWS_SECRET_ACCESS_KEY=<your-secret-key>
export BACKEND_API_URL=http://localhost:8000
```

## Step 3: Deploy Backend API

### 3.1 Install Dependencies
```bash
pip install -r requirements.txt
```

### 3.2 Run FastAPI Server
```bash
cd backend
uvicorn api:app --host 0.0.0.0 --port 8000
```

Or on EC2, using a process manager like Gunicorn:
```bash
gunicorn -w 2 -k uvicorn.workers.UvicornWorker backend.api:app --bind 0.0.0.0:8000
```

## Step 4: Deploy Streamlit Frontend

### 4.1 Run Locally
```bash
streamlit run frontend/app.py
```

### 4.2 Deploy on EC2
```bash
# Install
pip install streamlit

# Run as a background service (example with systemd)
sudo nano /etc/systemd/system/streamlit-resume-coach.service
```

Add the following content:
```ini
[Unit]
Description=Streamlit Resume Coach
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/path/to/resume_llm_analyzer
ExecStart=/usr/local/bin/streamlit run frontend/app.py --server.port 8501
Restart=always

[Install]
WantedBy=multi-user.target
```

Then run:
```bash
sudo systemctl daemon-reload
sudo systemctl start streamlit-resume-coach
sudo systemctl enable streamlit-resume-coach
```

## Step 5: Access the Application

- Frontend: `http://<backend-ip>:8501`
- API Health Check: `curl http://<backend-ip>:8000/health`

## Monitoring & Scaling

### Monitor SageMaker Endpoint
```bash
aws cloudwatch get-metric-statistics \
  --namespace AWS/SageMaker \
  --metric-name ModelLatency \
  --dimensions Name=EndpointName,Value=resume-coach-endpoint \
  --start-time 2023-01-01T00:00:00Z \
  --end-time 2023-01-02T00:00:00Z \
  --period 300 \
  --statistics Average
```

### Scale the Endpoint
```bash
aws sagemaker update-endpoint \
  --endpoint-name resume-coach-endpoint \
  --endpoint-config-name resume-coach-endpoint-config-v2
```

## Cost Optimization

- Use spot instances for non-production workloads
- Implement auto-scaling for SageMaker endpoints
- Set up CloudWatch alarms for unexpected endpoint behavior
- Consider using smaller model variants for cost savings

## Troubleshooting

1. **Endpoint not responding**: Check SageMaker console for InService status
2. **Authentication errors**: Verify AWS credentials in `.env` or IAM role
3. **Memory errors**: Upgrade instance type (e.g., `ml.g4dn.2xlarge`)
4. **Timeout issues**: Increase timeout in frontend/app.py and backend/api.py

## Cleanup

To avoid unnecessary charges, delete the SageMaker endpoint when no longer needed:
```bash
aws sagemaker delete-endpoint \
  --endpoint-name resume-coach-endpoint \
  --region us-east-1

aws sagemaker delete-endpoint-config \
  --endpoint-config-name meta-llama-3-1-8b-instruct-config \
  --region us-east-1

aws sagemaker delete-model \
  --model-name meta-llama-3-1-8b-instruct \
  --region us-east-1
```
