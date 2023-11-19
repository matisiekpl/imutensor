FROM python:3.7.4
WORKDIR /app
COPY . /app
RUN pip3 install torch numpy tensorflow pandas matplotlib boto3 flask
EXPOSE 3199
CMD ["python", "train.py"]