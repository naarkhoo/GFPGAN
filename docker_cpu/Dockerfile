FROM python:3.7-slim

RUN apt-get update && \
    # utility
    apt-get install -y libgomp1 nano less tree procps wget \
    # OpenCV
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1-mesa-glx

RUN python -m pip install --upgrade pip

RUN pip install --no-cache-dir torch -f https://torch.kmtea.eu/whl/stable.html

RUN mkdir /gfpgan/
WORKDIR /gfpgan/

COPY requirements.txt .
RUN pip install --no-cache-dir -r /gfpgan/requirements.txt

WORKDIR /gfpgan/api/

EXPOSE 5000
CMD bash run_api.sh
