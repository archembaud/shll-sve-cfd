import boto3
import time
from datetime import date

# Calculate today's date as DD-MM-YY
today = date.today()
human_readable_date = today.strftime("%d-%m-%y")

# Open an S3 client and move the file using the date as a prefix
s3 = boto3.resource('s3')
s3.meta.client.upload_file('./results.dat', 'sve-results', f'{human_readable_date}/results.dat')