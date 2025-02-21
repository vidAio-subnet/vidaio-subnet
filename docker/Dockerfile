FROM python:3.9-slim

WORKDIR /app

COPY ../requirements/common_requirements.txt /app/

RUN pip install --no-cache-dir -r common_requirements.txt

COPY ../services/ /app/services/

CMD ["python", "--version"]