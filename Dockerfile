# FROM python:3.10

# ADD main.py .
# ADD requirements.txt .
# RUN pip install -r requirements.txt

# CMD ["uvicorn", "main:app", "--reload"]

# criace-apis-image


# Build container
# FROM python:3.10-slim-buster

FROM python:3.10 AS builder

WORKDIR /cricace-app

ENV CUDA_VISIBLE_DEVICES=''
ENV TF_ENABLE_TENSORRT=0
ENV TF_DISABLE_DEPRECATION_WARNINGS=1
ENV TF_ENABLE_ONEDNN_OPTS=0

RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0


COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt


RUN pip install tensorflow-cpu


# Runtime container
FROM python:3.10-slim-buster
COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/

COPY . ./app


CMD ["python", "./app/main.py"]
# CMD ["uvicorn", "main:app", "--reload"]
