# Use Nvidia CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 as base

# Prevents prompts from packages asking for user input during installation
ENV DEBIAN_FRONTEND=noninteractive
# Prefer binary wheels over source distributions for faster pip installations
ENV PIP_PREFER_BINARY=1
# Ensures output from python is printed immediately to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# Install Python, git and other necessary tools
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    wget \
    libglib2.0-0 \
    build-essential \
    libgl1

# Clean up to reduce image size
RUN apt-get autoremove -y && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Clone ComfyUI repository
RUN git clone https://github.com/comfyanonymous/ComfyUI.git /comfyui

# Change working directory to ComfyUI
WORKDIR /comfyui

# Install ComfyUI dependencies
RUN pip3 install -r requirements.txt


# Install WAS Node
WORKDIR /comfyui/custom_nodes
RUN git clone https://github.com/WASasquatch/was-node-suite-comfyui.git
WORKDIR /comfyui/custom_nodes/was-node-suite-comfyui
RUN pip3 install -r requirements.txt
WORKDIR /comfyui/custom_nodes

# Install FacePasing
RUN git clone https://github.com/Ryuukeisyou/comfyui_face_parsing.git

# Install IPAdapter
RUN git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

# Install ComfyUI Various
RUN git clone https://github.com/jamesWalker55/comfyui-various

# Install InstantID
RUN git clone https://github.com/cubiq/ComfyUI_InstantID.git

RUN pip3 install transformers==4.26.1
#RUN pip3 install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install runpod requests ultralytics opencv-contrib-python xformers insightface==0.7.3 onnxruntime-gpu==1.16.2

###### NOBG
# Install Impact Pack
RUN git clone https://github.com/ltdrdata/ComfyUI-Impact-Pack.git
WORKDIR /comfyui/custom_nodes/ComfyUI-Impact-Pack
RUN pip3 install --no-cache-dir -r requirements.txt
RUN python3 install.py
WORKDIR /comfyui/custom_nodes

# Install ControlNet aux
RUN git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git
WORKDIR /comfyui/custom_nodes/comfyui_controlnet_aux
RUN pip3 install --no-cache-dir -r requirements.txt
WORKDIR /comfyui/custom_nodes

# Install Masquerade Nodes
RUN git clone https://github.com/BadCafeCode/masquerade-nodes-comfyui.git
###### NOBG

# Go back to the root
WORKDIR /comfyui

# Copy models
COPY checkpoints/sdxl/juggernautXL_v9Rundiffusionphoto2.safetensors /comfyui/models/checkpoints/
COPY checkpoints/controlnet/instantid/diffusion_pytorch_model.safetensors /comfyui/models/controlnet/instantid/
COPY checkpoints/controlnet/diffusers_xl_canny_full.safetensors /comfyui/models/controlnet/
COPY checkpoints/instantid/ip-adapter.bin /comfyui/models/instantid/
COPY checkpoints/blip/model_base_vqa_capfilt_large.pth /comfyui/models/blip/checkpoints/
COPY checkpoints/face_parsing/* /comfyui/models/face_parsing/
COPY checkpoints/ipadapter/ip-adapter-plus-face_sdxl_vit-h.safetensors /comfyui/models/ipadapter/
COPY checkpoints/ipadapter/ip-adapter-plus_sdxl_vit-h.safetensors /comfyui/models/ipadapter/
COPY checkpoints/clip_vision/CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors /comfyui/models/clip_vision/
COPY checkpoints/ultralytics/face_yolov8m.pt /comfyui/models/ultralytics/bbox/
COPY checkpoints/insightface/models/antelopev2/* /comfyui/models/insightface/models/antelopev2/
COPY checkpoints/insightface/inswapper_128.onnx /comfyui/models/insightface/

# Copy the prompt database and workflows
COPY ./prompts_db.json /comfyui/
COPY ./workflows/*.json /comfyui/workflows/

# Add the start and the handler
COPY src/start.sh src/rp_handler.py test_input.json /
RUN chmod +x /start.sh

# Start the container
CMD /start.sh
