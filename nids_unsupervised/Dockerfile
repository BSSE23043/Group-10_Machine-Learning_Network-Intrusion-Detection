FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --default-timeout=300 \
    --index-url https://pypi.org/simple/ \
    -r requirements.txt
EXPOSE 5000
CMD ["python", "app.py"]