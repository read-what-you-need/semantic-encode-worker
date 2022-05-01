import os
import time
import shutil
import glob
import re
import math
import scipy.spatial
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from collections import Counter, OrderedDict 

import requests

import io
import json
import pickle
import random

import datetime

import boto3
from kafka import KafkaConsumer
from kafka import KafkaProducer
from kafka.errors import KafkaError
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway, Counter
from utils import helper_functions 


# initializing environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region_name = os.getenv('AWS_REGION_NAME')

environment_type = os.getenv('env')
device_type = os.getenv('device')

if environment_type == "dev":
    model_path= './models/distilroberta-base-msmarco-v2'
if environment_type == "prod":
    model_path= 'distilroberta-base-msmarco-v2'
print('device-type:', device_type)

# deserializers
stringDeserializer = lambda m: m.decode('utf-8')

# monitoring
registry = CollectorRegistry()
monitoring_push_gate = os.getenv('MONITORING_PUSH_GATE')
signal_error_gauge = Gauge('semantic_job_failure', 'If 1 Indicates service is not up', registry=registry)
app_running_status_gauge = Gauge('semantic_job_app_status', 'If value between 5 and 20 indicatesIndicates service up', registry=registry)
app_time_to_process_file = Gauge('semantic_job_file_process_duration', 'Duration taken to process file', registry=registry)
app_request_received = Counter('semantic_job_request_recieved', 'Total requests recieved', registry=registry)
app_successfully_finished = Counter('semantic_job_request_success', 'Total succesful requests', registry=registry)

class PythonPredictor:

    def __init__(self):

        # download the information retrieval model trained on MS-MARCO dataset
        self.embedder = SentenceTransformer(model_path, device=device_type)
        
        # set the environment variables
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region_name = os.getenv('AWS_REGION_NAME')
        self.app_name= os.getenv('APP_NAME')
                
        self.BUCKET_NAME= os.getenv('AWS_FILES_BUCKET_NAME');

        self.kafka_broker_host=os.getenv('KAFKA_BROKER_HOST');
        self.kafka_group_id = os.getenv('KAFKA_GROUP_ID');
        self.kafka_consumer_topic_name = os.getenv('KAFKA_CONSUMER_TOPIC_NAME');
        self.kafka_producer_topic_name = os.getenv('KAFKA_PRODUCER_TOPIC_NAME');
        self.kafka_producer_status_update_topic_name =os.getenv('KAFKA_PRODUCER_STATUS_UPDATE_TOPIC_NAME')


        # establish connection with s3 BUCKET
        try:  
            self.s3 = boto3.client('s3', aws_access_key_id=self.aws_access_key_id , 
            aws_secret_access_key=self.aws_secret_access_key, 
            region_name=self.aws_region_name)

            self.s3_resource_upload = boto3.resource('s3' ,aws_access_key_id=self.aws_access_key_id , 
            aws_secret_access_key=self.aws_secret_access_key, 
            region_name=self.aws_region_name)

            print('Connected to s3 BUCKET!')
        except Exception as ex:
            print('\n\naws client error:', ex)
            signal_error_gauge.set(1)
            push_to_gateway(monitoring_push_gate , job=self.app_name, registry=registry)
            exit('Failed to connect to s3 BUCKET, terminating.')


        # establish connection with kafka broker
        try:
            self.kafka_consumer = KafkaConsumer(self.kafka_consumer_topic_name,
                         group_id=self.kafka_group_id,
                         bootstrap_servers=[self.kafka_broker_host],
                         key_deserializer= stringDeserializer,
                         auto_offset_reset='earliest',
                         max_poll_interval_ms=100,
                         value_deserializer=stringDeserializer)
            print('kakfa consumer connected')

            self.kafka_producer = KafkaProducer(bootstrap_servers=[self.kafka_broker_host])
            print('kakfa producer connected')

        except Exception as ex:
            print('\n\nkafka client error:', ex)
            signal_error_gauge.set(1)
            push_to_gateway(monitoring_push_gate , job=self.app_name, registry=registry)
            exit('Failed: Consumer/producer unable to connect to kafka broker, terminating.')


        # create temp dir for storing embeddings 
        self.dir = 'v2'

        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir)
        app_running_status_gauge.set(random.randint(5,20))
        push_to_gateway(monitoring_push_gate , job=self.app_name, registry=registry)    

        for message in self.kafka_consumer:
            app_running_status_gauge.set(random.randint(5,20))
            app_request_received.inc()
            push_to_gateway(monitoring_push_gate , job=self.app_name, registry=registry)

            print ("recieved message in topic: %s: key=%s value=%s" % (message.topic, message.key, message.value))
            self.uuid = message.value
            self.status_update_emit(self.uuid, "File processing started by heavy cruncher!")

            # process the message
            print('\njob request for UUID '+ self.uuid)

            # call worker process
           
            self.worker_job_encode(self.uuid)

            # invoke producer to signal job finished
            self.status_update_emit(self.uuid, "File processing finished.")
            self.producer_emit_job_finished_event(self.uuid)
            app_successfully_finished.inc()
            push_to_gateway(monitoring_push_gate , job=self.app_name, registry=registry)
            print('-----------------------------------------')

    @app_time_to_process_file.time()
    def worker_job_encode(self, uuid):

        try:
            os.mkdir('v2/'+uuid)

        except Exception as e:
            pass

        # download text file for processing
        print('\n\n✍️ downloading the text file ✍️')
        self.s3.download_file(self.BUCKET_NAME, 'v2/'+uuid+'/file.txt', 'v2/'+uuid+'/file.txt')

        with open('v2/'+uuid+'/file.txt', 'r') as file:
            file_list = file.read()

        top_n_dict = helper_functions.top_n_words(file_list)
            
        print('cleaning up the file a bit 👀')
        corpus = helper_functions.payload_text_preprocess(file_list)

        # heavy bottle neck, magic happes here
        print(' ⭐ processing the text file ⭐ ')
        corpus_embeddings = self.embedder.encode(corpus, show_progress_bar=True)

        # save embeddings to s3
        save_path = os.path.join('v2', uuid, 'corpus_encode.npy')
        pickle_byte_obj = pickle.dumps(corpus_embeddings)
        print('job almost finished 😁, uploading to 🌥️ ')
        self.s3_resource_upload.Object(self.BUCKET_NAME, 'v2/'+uuid+'/topNwords.json').put(Body=json.dumps(top_n_dict, indent=1)) 
        self.s3_resource_upload.Object(self.BUCKET_NAME, save_path).put(Body=pickle_byte_obj)

        with app_time_to_process_file.time():
            pass

    def producer_emit_job_finished_event(self, uuid):

        # Asynchronous by default
        future = self.kafka_producer.send(self.kafka_producer_topic_name, bytes(uuid, 'utf-8'))

        # Block for 'synchronous' sends
        try:
            record_metadata = future.get(timeout=5)
        except KafkaError:
            # Decide what to do if produce request failed...
            print("kafka_producer send failed for uuid:", uuid)
            pass

        # Successful result returns assigned partition and offset
        print("message sent to topic:%s partition:%s offset:%s:" % (record_metadata.topic, record_metadata.partition, record_metadata.offset))        

    def status_update_emit(self, uuid, status):

        future = self.kafka_producer.send(self.kafka_producer_status_update_topic_name, key=bytes(uuid, 'utf-8'), value=(bytes(status, 'utf-8')))
        try:
            record_metadata = future.get(timeout=60)
        except KafkaError:
            print("kafka_producer status_update_emit send failed for uuid:", uuid)
            pass

        # Successful result returns assigned partition and offset
        print("message sent to topic:%s partition:%s offset:%s:" % (record_metadata.topic, record_metadata.partition, record_metadata.offset))        


PythonPredictor()