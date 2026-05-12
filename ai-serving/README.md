# AI Serving
This is the image creation for the RAG models used in the cli tooling. This is to reduce the loading and unloading of the Large Language Model and Clip embedding model.
The models are deployed and served 


## Getting Started
### Requirements
This project requires docker and docker-compose to be installed in order to build and deploy these models.
If you want to use a GPU this requires the nvidia-container-toolkit package.

### Building and Deploying
To begin, store the models weights in the data folder and build the docker image
```bash
docker build -t localhost/rag-models:v1 .
```

Once built, update the docker-compose.yml to match the version tag that was just built and deploy the service with docker compose
```bash
docker compose up --detach

# To check logs to ensure it has been successfully loaded
docker compose logs
```

Once deployed, update the environment variables to
```bash
export MODEL_SERVING="DEPLOYED"
export MODEL_ENDPOINT="http://localhost:8000"
export MODEL_GENERATE_CONTENT="/gemma3/generate-content/"
export MODEL_IMAGE_EMBED="/clip/embed-image/"
export MODEL_IMAGE_SERACH="/clip/image-search/"
```
Alternatively, create a .env file in the root directory of rag-search-engine with these values stored in it
