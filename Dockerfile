FROM python:3.10

RUN apt-get update -y && \
    apt-get install python3-opencv -y 

WORKDIR /home/src

COPY . /home/src/

RUN pip install -r requirements.txt

CMD ["python", "main.py"]