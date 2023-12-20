#!/bin/bash
ID=$(openssl rand -hex 12)
echo "$ID"
docker build -t matisiekpl/imutensor:"$ID" --platform linux/amd64 .
docker push matisiekpl/imutensor:"$ID"
kubectl set image deployments/imutensor api=matisiekpl/imutensor:"$ID"
