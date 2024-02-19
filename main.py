#-*- coding: utf-8 -*-
import torch
from transformers import pipeline
import pandas as pd
import os
from datasets import Dataset, DatasetDict
import datasets
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import boto3
from botocore.exceptions import ClientError
import logging
import warnings
from io import StringIO
from wsgiref.simple_server import make_server
from pyramid.config import Configurator
from pyramid.response import Response

warnings.filterwarnings("ignore", message="Length of IterableDataset.*")

device = "cuda:0" if torch.cuda.is_available() else "cpu"

train_df = pd.DataFrame(columns = ["cont", "audio", "Resultado"])

AWS_ACCESS_KEY = "AKIARQECLQAX7SKFACZF"
AWS_SECRET_ACCESS_KEY = "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"
AWS_S3_BUCKET_NAME = "bucket-prueba1coes"
AWS_REGION_NAME = "sa-east-1"

s3_client = boto3.resource(
        service_name = 's3',
        region_name= AWS_REGION_NAME,
        aws_access_key_id= AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )

s3 = boto3.client('s3',
                                region_name= AWS_REGION_NAME,
                                aws_access_key_id= AWS_ACCESS_KEY,
                                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
                                )

transcribe = pipeline(
                      task            = "automatic-speech-recognition",
                      model           = "model/",
                      chunk_length_s  = 30,
                      device          = device
                      )
transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(language="es", task="transcribe")

def main():
    global train_df
    # Upload the file
    
    
    bukcet = s3_client.Bucket(AWS_S3_BUCKET_NAME)

    cont = 0
    for file in bukcet.objects.all():
        if file.key.endswith(".opus"):

            train_df = pd.concat([train_df, pd.DataFrame({'cont': [cont], 'audio': [file.key]})], ignore_index=True)
            
            response = s3.get_object(Bucket = AWS_S3_BUCKET_NAME, Key = file.key)
            audio_data = response['Body'].read()

            train_df.loc[cont, "Resultado"] = transcribe(audio_data)["text"]
            cont = cont + 1
            
    csv_buffer = StringIO()
    train_df.to_csv(csv_buffer, index=False, encoding='utf-8')   
    csv_buffer.seek(0)

    response = s3.put_object(Body= csv_buffer.getvalue(), Bucket = AWS_S3_BUCKET_NAME, Key = "Trancripcion.csv")

    print(train_df)

    warnings.resetwarnings()

def hello_world(request):
    name = os.environ.get('NAME')
    if name == None or len(name) == 0:
        name = "world"
    message = "Hello, " + name + "!\n"
    return Response(message)

if __name__ == "__main__":
    main()
    port = int(os.environ.get("PORT"))
    with Configurator() as config:
        config.add_route('hello', '/')
        config.add_view(hello_world, route_name='hello')
        app = config.make_wsgi_app()
    server = make_server('0.0.0.0', port, app)
    server.serve_forever()
