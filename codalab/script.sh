#!/bin/bash

cl upload brain.py
cl upload hyperparameters.py
cl upload run.py
cl upload agent.py
cl upload environment.py
cl upload memory.py

cl run :brain.py :hyperparameters.py :run.py :agent.py :environment.py :memory.py 'python3 run.py' -n sort-run --request-docker-image prabhjotrai/openai-gym:v1
