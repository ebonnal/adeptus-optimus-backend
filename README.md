# Adeptus Optimus Backend

|branch|CI|CD|
|--|--|--|
|main|[![Actions Status](https://github.com/bonnal-enzo/adeptus-optimus-backend/workflows/test/badge.svg?branch=main)](https://github.com/bonnal-enzo/adeptus-optimus-backend/actions)[![Actions Status](https://github.com/bonnal-enzo/adeptus-optimus-backend/workflows/build/badge.svg?branch=main)](https://github.com/bonnal-enzo/adeptus-optimus-backend/actions)|[![Actions Status](https://github.com/bonnal-enzo/adeptus-optimus-backend/workflows/deploy/badge.svg?branch=main)](https://github.com/bonnal-enzo/adeptus-optimus-backend/actions)|
|prod|[![Actions Status](https://github.com/bonnal-enzo/adeptus-optimus-backend/workflows/test/badge.svg?branch=prod)](https://github.com/bonnal-enzo/adeptus-optimus-backend/actions)[![Actions Status](https://github.com/bonnal-enzo/adeptus-optimus-backend/workflows/build/badge.svg?branch=prod)](https://github.com/bonnal-enzo/adeptus-optimus-backend/actions)|[![Actions Status](https://github.com/bonnal-enzo/adeptus-optimus-backend/workflows/deploy/badge.svg?branch=prod)](https://github.com/bonnal-enzo/adeptus-optimus-backend/actions)|




# Locally
## Run app
```bash
pip3 install -r requirements.txt

cd src
<<<<<<< HEAD
=======

>>>>>>> 43f4f51b4e325b94ed1c96f14bef6c1aa0eaa5ec
python app.py
# or
python -m flask run --port=8080
# or
waitress-serve --call 'app:create_app'
```

and 

```bash
curl -X GET http://127.0.0.1:8080/engine/
```

and 

```bash
curl -X GET http://127.0.0.1:5000/engine/
```

## Run unit tests
```bash
cd src
python -m unittest
```

# GCP Deployment
CD deploys [the app on Cloud Run](https://console.cloud.google.com/run/detail/europe-west1/engine/metrics?authuser=0&project=adeptus-optimus)


# Locally
## Run app
```bash
pip3 install -r requirements.txt

python app.py
# or
python -m flask run
```

## Run unit tests
```bash
cd src
python -m unittest
```

# GCP Deployment
CD deploys [the app on Cloud Run](https://console.cloud.google.com/run/detail/europe-west1/engine/metrics?authuser=0&project=adeptus-optimus)
