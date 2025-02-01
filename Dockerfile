FROM python:3.11.2

# Instala dependencias del sistema (incluyendo Xvfb)
# Instala Xvfb y dependencias gráficas
RUN apt-get update -y && \
    apt-get install -y \
    python3-tk \
    x11-utils

WORKDIR /home/src

COPY . /home/src/

RUN pip install -r requirements.txt

# Configura el DISPLAY para apuntar al host de Windows
ENV DISPLAY=host.docker.internal:0.0
# Inicia Xvfb y ejecuta la aplicación
CMD ["python", "main.py"]