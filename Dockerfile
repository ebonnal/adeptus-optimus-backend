# syntax=docker/dockerfile:1
FROM ubuntu:18.04
RUN apt-get update
RUN apt-get -y install python3-pip
RUN pip3 install requests numpy flask
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
COPY src/adeptus_optimus_backend /app/adeptus_optimus_backend
COPY src/app.py /app/app.py
WORKDIR /app
CMD python3 -m flask run --host=0.0.0.0 --port=8080
