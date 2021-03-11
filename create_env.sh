#!/bin/bash
> .env
echo 'AWS_ACCESS_KEY_ID='$AWS_ACCESS_KEY_ID >> .env
echo 'AWS_SECRET_ACCESS_KEY='$AWS_SECRET_ACCESS_KEY >> .env
echo 'AWS_REGION_NAME='$AWS_REGION_NAME >> .env
echo 'MONGO_URI='$MONGO_URI >> .env