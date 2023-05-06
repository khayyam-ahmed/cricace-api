# FROM python:3.10

# ADD main.py .
# ADD requirements.txt .
# RUN pip install -r requirements.txt

# CMD ["uvicorn", "main:app", "--reload"]

# criace-apis-image

FROM python:3.10-slim-buster

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--reload"]
