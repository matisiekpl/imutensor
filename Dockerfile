FROM python:3.7.4 as trainer
WORKDIR /app
COPY . /app
RUN pip3 install torch numpy pandas matplotlib flask
RUN python train.py
EXPOSE 3199
CMD ["python", "serve.py"]