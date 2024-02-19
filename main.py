#-*- coding: utf-8 -*-
import torch
from transformers import pipeline
import pandas as pd
import os
import numpy
from datasets import Dataset, DatasetDict
import datasets
from transformers.pipelines.pt_utils import KeyDataset
from tqdm.auto import tqdm
import warnings
import boto3
from botocore.exceptions import ClientError
import logging
from io import StringIO

print("Se importaron los módulos")
warnings.filterwarnings("ignore", message = "Length of IterableDataset.*")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
AWS_ACCESS_KEY          = "AKIARQECLQAX7SKFACZF"
AWS_SECRET_ACCESS_KEY   = "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"
AWS_S3_BUCKET_NAME      = "bucket-prueba1coes"
AWS_REGION_NAME         = "sa-east-1"

print("Previo al asignar el modelo")
transcribe = pipeline(
                        task = "automatic-speech-recognition",
                        model = "Cristhian2430/whisper-large-coes-v3",
                        chunk_length_s = 30,
                        device = device
                        )
print("Se descargo modelo")
transcribe.model.config.forced_decoder_ids = transcribe.tokenizer.get_decoder_prompt_ids(
    language = "es", 
    task = "transcribe"
    )
print("Se ajusto modelo")
s3_client   = boto3.resource(
                            service_name            = "s3",
                            region_name             = AWS_REGION_NAME,
                            aws_access_key_id       = AWS_ACCESS_KEY,
                            aws_secret_access_key   = AWS_SECRET_ACCESS_KEY
                            )
s3          = boto3.client("s3",
                            region_name             = AWS_REGION_NAME,
                            aws_access_key_id       = AWS_ACCESS_KEY,
                            aws_secret_access_key   = AWS_SECRET_ACCESS_KEY
                            )

bucket = s3_client.Bucket(AWS_S3_BUCKET_NAME)
print("Se conectaron")
df_res = pd.DataFrame(columns = ["cont", "audio", "resultado"])
cont = 0
print("Inicia Bucle Transcripción")
for file in bucket.objects.all():
    if file.key.endswith(".opus"):
        print("-------------------------")
        print("Se encontró archivo OPUS")
        df_res = pd.concat([df_res, pd.DataFrame({"cont":[cont], "audio":[file.key]})], ignore_index = True)
        print("Se agregó al DataFrame")
        response = s3.get_object(Bucket = AWS_S3_BUCKET_NAME, Key = file.key)
        print("Se obtuvo archivo")
        audio_data = response["Body"].read()
        df_res.loc[cont, "resultado"] = transcribe(audio_data)["text"]
        print("Se transcribió")
        cont = cont + 1
print("Terminó Bucle Transcripción")
csv_buffer = StringIO()
df_res.to_csv(csv_buffer, index = False, encoding = "utf-8")
csv_buffer.seek(0)
print("Se procederá a enviar el csv a S3")
response = s3.put_object(Body = csv_buffer.getvalue(), Bucket = AWS_S3_BUCKET_NAME, Key = "Transcripcion.csv")

warnings.resetwarnings()

df_1 = pd.DataFrame({"Col1":[1,2], "Col2":[3,4]})

def handler(event, context):
    return {"statusCode":200, "body": df_1.loc[1].tolist()}