FROM python:3.9
WORKDIR /app
COPY ./ptcr /app/ptcr
COPY main.py /app

RUN pip install -r /app/ptcr/requirements.txt

CMD ["python", "/app/main.py"]
