import base64
import json
import importlib
import requests

# read input image
image = './test_images/image1.jpg'
with open(image, 'rb') as file:
    encoded = base64.b64encode(file.read())

port = YOUR_PORT
scoring_url = 'http://127.0.0.1:{}/score'.format(port)
# prepared the payload and header
payload = []
payload.append("{}".format(encoded))
req_str = json.dumps(payload)
headers = {'Content-Type': 'application/json'}

# score the images using the REST API
res = requests.post(scoring_url, data=req_str, headers=headers)
print(res.text)
