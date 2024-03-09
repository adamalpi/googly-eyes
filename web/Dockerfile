FROM python:3.11

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH "$PATH:/root/.local/bin/"

RUN mkdir /service
WORKDIR /service
ADD . /service/

RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-dev

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]

# docker build -t web-service:latest .
# docker run --rm --name=web-service -p 8501:8501 web-service