FROM continuumio/miniconda3

WORKDIR /app

# Copiar environment.yml
COPY . ./
# Crear el entorno con Python 3.10.13 (¡ajusta la versión en environment.yml si es necesario!)
RUN conda env create -f environment.yml --force

# Inicializar conda para bash (¡CRUCIAL!)
RUN conda init bash

# Activar el entorno (usando conda run para evitar problemas de shell)
RUN conda run -n myenv echo "Environment activated"  # Solo para verificar

# Copiar el código de la aplicación

# Hacer entrypoint.sh ejecutable
RUN chmod +x entrypoint.sh

# ENTRYPOINT (ejecuta entrypoint.sh)
ENTRYPOINT ["./entrypoint.sh"]