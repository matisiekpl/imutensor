FROM pytorch/pytorch
WORKDIR /app
COPY . /app
RUN pip install torch numpy tensorflow pandas matplotlib boto3 flask
EXPOSE 3199
CMD ["python", "train.py"]