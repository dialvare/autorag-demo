# Usamos la imagen oficial y segura de Red Hat (Python 3.12)
FROM registry.access.redhat.com/ubi9/python-312:latest

# Cambiamos al directorio de trabajo del contenedor
WORKDIR /opt/app-root/src

# Copiamos los archivos
COPY requirements.txt .
COPY app.py .

# Instalamos las librerías de forma limpia
RUN pip install --no-cache-dir -r requirements.txt

# Exponemos el puerto estándar de Streamlit
EXPOSE 8501

# Comando de arranque sin florituras
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]