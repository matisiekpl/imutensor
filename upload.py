import os
import boto3
from botocore.exceptions import NoCredentialsError

bucket = 'mateuszwozniak'

session = boto3.Session(
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
)


def upload_to_s3(local_file, bucket, s3_file):
    s3 = session.client('s3')
    try:
        s3.delete_object(Bucket=bucket, Key=s3_file)
        s3.upload_file(local_file, bucket, s3_file)
        s3.put_object_acl(Bucket=bucket, Key=s3_file, ACL='public-read')
        print(f"{local_file} uploaded successfully to {bucket}/{s3_file}")
    except FileNotFoundError:
        print(f"The file {local_file} was not found")
    except NoCredentialsError:
        print("Credentials not available")


if __name__ == "__main__":
    aws_access_key_id = 'YOUR_ACCESS_KEY_ID'
    aws_secret_access_key = 'YOUR_SECRET_ACCESS_KEY'
    upload_to_s3('model.h5', bucket, 'model.h5')
    upload_to_s3('info.json', bucket, 'info.json')
