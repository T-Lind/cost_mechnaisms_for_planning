FROM python:3.9

WORKDIR /app

COPY ./ptcr2 /app/ptcr2
COPY main.py /app
COPY requirements.txt /app
# NOTE: ENSURE ALL FILES YOU WANT TO COPY ARE COPIED ABOVE. THIS ENSURES TEST/DEV FILES ARE NOT COPIED TO PRODUCTION

RUN pip install -r /app/requirements.txt

CMD ["python", "/app/main.py"]
