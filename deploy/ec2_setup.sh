#!/usr/bin/env bash
# ec2_setup.sh — Run this directly on the EC2 terminal to deploy Resume Coach.
# No Docker required. Runs Streamlit as a background process.
#
# Usage:
#   bash deploy/ec2_setup.sh

set -e

APP_DIR="/home/ec2-user/resume-coach"
REPO="https://github.com/Ramachandra89/resume_llm_analyzer.git"
ENDPOINT="jumpstart-dft-llama-3-1-8b-instruct-20260417-030131"
REGION="us-east-1"

echo "=== Step 1: Stop and remove all Docker containers ==="
docker stop $(docker ps -q) 2>/dev/null && echo "Containers stopped" || echo "No running containers"
docker rm   $(docker ps -aq) 2>/dev/null && echo "Containers removed" || echo "Nothing to remove"

echo ""
echo "=== Step 2: Kill any existing Streamlit process ==="
pkill -f "streamlit run" 2>/dev/null && echo "Old Streamlit killed" || echo "No existing Streamlit"

echo ""
echo "=== Step 3: Install / upgrade Python packages ==="
pip3 install --quiet --upgrade streamlit boto3 PyPDF2 python-dotenv

echo ""
echo "=== Step 4: Clone or update the repo ==="
if [ -d "$APP_DIR/.git" ]; then
    echo "Repo exists — pulling latest..."
    cd "$APP_DIR" && git pull
else
    echo "Cloning repo..."
    git clone "$REPO" "$APP_DIR"
    cd "$APP_DIR"
fi

echo ""
echo "=== Step 5: Write .env ==="
cat > "$APP_DIR/.env" <<ENV
SAGEMAKER_ENDPOINT_NAME=${ENDPOINT}
AWS_REGION=${REGION}
ENV

echo ".env written (AWS credentials come from the EC2 IAM role)"

echo ""
echo "=== Step 6: Start Streamlit on port 8501 ==="
cd "$APP_DIR"
nohup python3 -m streamlit run streamlit_app.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    > /tmp/streamlit.log 2>&1 &

sleep 3
if pgrep -f "streamlit run" > /dev/null; then
    PUBLIC_IP=$(curl -s --max-time 3 http://169.254.169.254/latest/meta-data/public-ipv4 || echo "<your-ec2-ip>")
    echo ""
    echo "✅  Resume Coach is running!"
    echo "    Open: http://${PUBLIC_IP}:8501"
    echo "    Logs: tail -f /tmp/streamlit.log"
else
    echo "❌  Streamlit failed to start. Check logs:"
    cat /tmp/streamlit.log
fi
