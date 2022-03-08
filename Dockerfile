# syntax=docker/dockerfile:1
FROM python:3.9-slim-bullseye
COPY requirements.txt /tmp
RUN python -m pip install -r /tmp/requirements.txt
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
COPY src/adeptus_optimus_backend /app/adeptus_optimus_backend
COPY src/app.py /app/app.py
WORKDIR /app
CMD waitress-serve --threads=4 --call 'app:create_app'
