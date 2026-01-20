FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY health_check.py .
COPY Weather_Models/ ./Weather_Models/

EXPOSE 8501

CMD python health_check.py && streamlit run app.py --server.address=0.0.0.0 --server.port=8501