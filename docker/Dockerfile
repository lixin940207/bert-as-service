FROM tensorflow/tensorflow:1.12.0-gpu-py3
RUN pip install bert-serving-server
COPY ./ /app/
COPY ./docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh
WORKDIR /app
ENTRYPOINT ["/app/entrypoint.sh"]
CMD []

