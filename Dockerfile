# Use the PyTorch base image
FROM runpod/pytorch:1.0.3-cu1290-torch290-ubuntu2204

# Set working directory
WORKDIR /

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI application code
# (Assuming your modified code is named main.py)
COPY main.py .

# Expose the port (RunPod load balancers usually look for 8000 or 80)
EXPOSE 8000

# Start uvicorn with workers=1 (GPU tasks are usually compute-bound,
# so one worker per pod is often more stable for LLMs)

CMD ["python", "-u", "main.py"]
