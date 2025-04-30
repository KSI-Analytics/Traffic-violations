#!/bin/bash

echo "Installing requirements..."
pip install -r requirements.txt

# Run gunicorn
gunicorn app:server --workers 4 --bind=0.0.0.0:8000




