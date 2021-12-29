FROM python:3.7-slim

RUN apt-get update && \
    apt-get install -y libgomp1 nano less tree procps wget \
    # OpenCV
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir torch -f https://torch.kmtea.eu/whl/stable.html

RUN mkdir /gfpgan/
WORKDIR /gfpgan/

# weights
RUN wget https://github.com/TencentARC/GFPGAN/releases/download/v0.2.0/GFPGANCleanv1-NoCE-C2.pth \
        -P experiments/pretrained_models/ &&\
    wget https://github.com/TencentARC/GFPGAN/releases/download/v0.1.0/GFPGANv1.pth \
        -P experiments/pretrained_models/ &&\
    wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth \
        -P experiments/pretrained_models/ &&\
    wget https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth \
        -P /usr/local/lib/python3.7/site-packages/facexlib/weights

COPY requirements.txt .
RUN pip install --no-cache-dir -r /gfpgan/requirements.txt

WORKDIR /gfpgan/api/

EXPOSE 5000
CMD bash run_api.sh
