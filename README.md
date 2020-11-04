# Adeptus Optimus Backend v1

## Run flask app locally
In root:
`python app.py`

## Test
In root:
`python -m unittest`

## Deployment notes
Using a 200MHz / 128MB **single** instance:
- One day DDOS at 10 weapons
  - thanks to min exec time of 3sec, it will need one day to 
  reach the free 5Go egress bandwidth limit. 
  - 5 additional gigs cost 0.12*5 = 0.60 cents