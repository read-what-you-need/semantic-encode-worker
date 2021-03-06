# this is an example for cortex release 0.21 and may not deploy correctly on other releases of cortex
import boto3
import os
import time
import shutil
import glob
import re
import math
import numpy as np
import scipy.spatial
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from collections import Counter, OrderedDict 

import json

import io
import numpy
import pickle

import pymongo
import datetime
from bson.objectid import ObjectId

from utils import helper_functions 


class PythonPredictor:

    def __init__(self, config):

        # download the information retrieval model trained on MS-MARCO dataset
        self.embedder = SentenceTransformer('distilroberta-base-msmarco-v2')
        
        # set the environment variables
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region_name = os.getenv('AWS_REGION_NAME')
        self.mongo_uri = os.getenv('MONGO_URI')

                
        self.QUEUE_NAME='readneed_encode_jobs.fifo'
        self.BUCKET='readneedobjects'

        # establish connection with s3 bucket
        try:  
            self.s3 = boto3.client('s3', aws_access_key_id=self.aws_access_key_id , 
            aws_secret_access_key=self.aws_secret_access_key, 
            region_name=self.aws_region_name)

            self.s3_resource_upload = boto3.resource('s3' ,aws_access_key_id=self.aws_access_key_id , 
            aws_secret_access_key=self.aws_secret_access_key, 
            region_name=self.aws_region_name)

            print('Connected to s3 bucket!')
        except Exception as ex:
            print('\n\naws client error:', ex)
            exit('Failed to connect to s3 bucket, terminating.')

        # establish connection with sqs client
        try:
            self.sqs_client = boto3.client('sqs', aws_access_key_id=self.aws_access_key_id , 
            aws_secret_access_key=self.aws_secret_access_key, 
            region_name=self.aws_region_name)

            print('Connected to sqs queue!')

        except Exception as ex:
            print('\n\naws sqs client error:', ex)
            exit('Failed to connect to sqs, terminating.')

        self.queues = self.sqs_client.list_queues(QueueNamePrefix=self.QUEUE_NAME) # we filter to narrow down the list
        self.test_queue_url = self.queues['QueueUrls'][0]

        # mongo collection client
        self.mongo_client = pymongo.MongoClient(self.mongo_uri)
        self.DB = self.mongo_client['readneed']
        self.files_collection = self.DB['files']
        
        # create temp dir for storing embeddings 
        self.dir = 'v2'

        if os.path.exists(self.dir):
            shutil.rmtree(self.dir)
        os.makedirs(self.dir)         


        while True:
            messages = self.sqs_client.receive_message(QueueUrl=self.test_queue_url,MaxNumberOfMessages=1, VisibilityTimeout=120) # adjust MaxNumberOfMessages if needed
            if 'Messages' in messages: # when the queue is exhausted, the response dict contains no 'Messages' key
                for message in messages['Messages']: # 'Messages' is a list
        
                    self.uuid = message['Body']
        
                    # process the messages
                    print('\njob request for UUID '+ self.uuid)
        
                    # call worker process
                    self.worker_job_encode(self.uuid)
        
                    # next, we delete the message from the queue so no one else will process it again
                    self.sqs_client.delete_message(QueueUrl=self.test_queue_url,ReceiptHandle=message['ReceiptHandle'])
                    
                    print('-----------------------------------------')
            else:
                print('Queue is now empty')
                time.sleep(30)


    def worker_job_encode(self, uuid):

        try:
            os.mkdir('v2/'+uuid)

        except Exception as e:
            pass


        # download text file for processing
        print('\n\n?????? downloading the text file ??????')
        self.s3.download_file(self.BUCKET, 'v2/'+uuid+'/text_content.txt', 'v2/'+uuid+'/text_content.txt')


        with open('v2/'+uuid+'/text_content.txt', 'r') as file:
            file_list = file.read()

        top_n_dict = helper_functions.top_n_words(file_list)
            


        print('cleaning up the file a bit ????')
        corpus = helper_functions.payload_text_preprocess(file_list)



        # heavy bottle neck, magic happes here
        print(' ??? processing the text file ??? ')
        corpus_embeddings = self.embedder.encode(corpus, show_progress_bar=True)

        # save embeddings to s3
        save_path = os.path.join('v2', uuid, 'corpus_encode.npy')
        pickle_byte_obj = pickle.dumps(corpus_embeddings)
        print('job almost finished ????, uploading to ??????? ')
        self.s3_resource_upload.Object(self.BUCKET, 'v2/'+uuid+'/topNwords.json').put(Body=json.dumps(top_n_dict, indent=1)) 
        self.s3_resource_upload.Object(self.BUCKET, save_path).put(Body=pickle_byte_obj)

        # update status in mongo collection
        self.update_query = { 'uuid' : uuid }
        self.new_value = { '$set': { 'processStatus': 'true' } }
        self.files_collection.update_one(self.update_query, self.new_value)


  

    def predict(self, payload):
        
        # extract values from the request payload
        
        # sess stores a file's uuid
        # a unique identifier to link to an uploaded file's text file, encodings and top words
        

        response = "worker online"
        
        return response
