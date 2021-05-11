# Adeptus Optimus Backend v1

## Run flask app locally
Install requirements
`pip3 install -r requirements.txt`

In root:
`python app.py`
or 
`python -m flask run`

## Test
In root:
`python -m unittest`

## Deployment notes
### As a Cloud Function
Using a 200MHz / 128MB **single** instance:
- One day DDOS at 10 weapons
  - thanks to min exec time of 3sec, it will need one day to 
  reach the free 5Go egress bandwidth limit. 
  - 5 additional gigs cost 0.12*5 = 0.60 cents
  
### As a Cloud Run service
  
## Push image `foo:bar` to Google Cloud Registry

[gcloud credential helper doc](https://cloud.google.com/container-registry/docs/advanced-authentication#gcloud-helper)
```bash
gcloud auth login
gcloud auth activate-service-account {service-account} --key-file={path to key file .json}
gcloud auth configure-docker
# sync config to be  able to use docker as root and find creds
sudo cp ~/.config/gcloud /root/.config/gcloud
sudo docker tag foo:bar eu.gcr.io/{project-id}/foo:bar
sudo docker push eu.gcr.io/{project-id}/foo:bar
```