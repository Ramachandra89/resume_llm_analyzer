# Deployment Guide for Resume Analysis with LLaMA-2-7b

This guide will help you set up and deploy the resume analysis system using LLaMA-2-7b on AWS.

## AWS Setup

1. Launch an EC2 Instance:
   - Instance Type: g4dn.xlarge (or better for GPU support)
   - AMI: Ubuntu Server 20.04 LTS
   - Storage: At least 50GB
   - Security Group: Allow inbound traffic on ports 22 (SSH) and 8000 (API)

2. Connect to your instance:
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-ip
   ```

3. Install NVIDIA drivers and CUDA:
   ```bash
   sudo apt-get update
   sudo apt-get install -y nvidia-driver-525
   sudo apt-get install -y nvidia-cuda-toolkit
   ```

4. Install Python and dependencies:
   ```bash
   sudo apt-get install -y python3.9 python3.9-venv
   python3.9 -m venv venv
   source venv/bin/activate
   ```

5. Clone the repository and install requirements:
   ```bash
   git clone <your-repo-url>
   cd resume_llm_analyzer
   pip install -r requirements.txt
   ```

## Model Deployment

1. Request access to LLaMA-2:
   - Visit https://ai.meta.com/llama/
   - Request access to LLaMA-2
   - Once approved, you'll receive a Hugging Face token

2. Set up environment variables:
   ```bash
   echo "HUGGINGFACE_TOKEN=your_token_here" > .env
   ```

3. Start the FastAPI server:
   ```bash
   uvicorn backend.api:app --host 0.0.0.0 --port 8000
   ```

## Testing the Deployment

1. Test the health endpoint:
   ```bash
   curl http://localhost:8000/health
   ```

2. Test the analysis endpoint:
   ```bash
   curl -X POST http://localhost:8000/analyze \
     -H "Content-Type: application/json" \
     -d '{"resume_text": "your resume text", "job_description": "job description"}'
   ```

## Monitoring and Maintenance

1. Monitor GPU usage:
   ```bash
   nvidia-smi
   ```

2. Check logs:
   ```bash
   tail -f /var/log/syslog
   ```

## Troubleshooting

1. If the model fails to load:
   - Check GPU memory usage
   - Verify CUDA installation
   - Check Hugging Face token validity

2. If the API is not responding:
   - Check if the server is running
   - Verify security group settings
   - Check application logs

## Security Considerations

1. Always use HTTPS in production
2. Implement rate limiting
3. Add authentication to the API
4. Regularly update dependencies
5. Monitor system resources 