FROM python:3.9
WORKDIR /app
COPY ./ptcr2 /app/ptcr2
COPY main.py /app
COPY requirements.txt /app

RUN pip install -r /app/requirements.txt

CMD ["python", "/app/main.py"]
