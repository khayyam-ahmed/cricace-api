

FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9-slim
# FROM python:3.10 AS builder

WORKDIR /cricace-app

ENV CUDA_VISIBLE_DEVICES=''
ENV TF_ENABLE_TENSORRT=0
ENV TF_DISABLE_DEPRECATION_WARNINGS=1
ENV TF_ENABLE_ONEDNN_OPTS=0

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0


COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install --upgrade pip && \
    pip install -r requirements.txt


RUN pip install tensorflow-cpu
RUN pip install ultralytics==8.0.20 


COPY . .


# CMD ["python", "./app/main.py"]
# CMD ["uvicorn", "main:app", "--reload"]
