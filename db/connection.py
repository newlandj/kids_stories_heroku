import os
import psycopg2
import boto3


def get_db_password_from_ssm(parameter_name: str) -> str:
    """
    Fetch the database password securely from AWS SSM Parameter Store.
    """
    ssm = boto3.client('ssm')
    response = ssm.get_parameter(Name=parameter_name, WithDecryption=True)
    return response['Parameter']['Value']


def get_db_connection():
    """
    Establish and return a psycopg2 connection to the PostgreSQL RDS instance.
    Environment variables required:
      - DB_HOST
      - DB_NAME
      - DB_USER
      - DB_PASSWORD_SSM_PARAM (the SSM parameter name for the DB password)
    """
    host = os.environ['DB_HOST']
    dbname = os.environ['DB_NAME']
    user = os.environ['DB_USER']
    password = os.environ['DB_PASSWORD']

    # SSM is not working in AWS, Will reimplement this some other time. 
    # password = get_db_password_from_ssm(os.environ['DB_PASSWORD_SSM_PARAM'])
    return psycopg2.connect(
        host=host,
        dbname=dbname,
        user=user,
        password=password,
        connect_timeout=10
    )
