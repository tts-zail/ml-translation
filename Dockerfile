# Use a pre-configured RunPod image
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /

# Copy requirements and install only the missing pieces
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your handler code
COPY handler.py .

# Start the handler
CMD [ "python", "-u", "/handler.py" ]