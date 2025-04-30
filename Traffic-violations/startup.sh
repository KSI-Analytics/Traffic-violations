#!/bin/bash

echo "Installing requirements..."
pip install -r requirements.txt

gunicorn app:server --bind=0.0.0.0 --timeout 600




