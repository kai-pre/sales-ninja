FROM python:3.10.6-buster

COPY requirements_light.txt /requirements.txt
RUN pip install -r requirements.txt

COPY salesninja /salesninja
COPY setup.py /setup.py

RUN pip install .

COPY gcs-key.json /gcs-key.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/gcs-key.json

CMD uvicorn salesninja.api.api:app --host 0.0.0.0 --port $PORT
