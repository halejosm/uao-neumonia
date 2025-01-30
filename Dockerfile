FROM python:3.10.13

RUN apt-get update -y && \
    apt-get install python3-opencv -y 

WORKDIR /home/src

COPY . ./
# Instalar dependencias específicas (incluyendo TensorFlow)
RUN pip cache purge
RUN pip install --no-cache-dir -r requirements.txt -v


# Definir el comando de inicio (ajusta según tu aplicación)
CMD ["python", "src/main.py"]