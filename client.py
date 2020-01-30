import json
import requests

api_url = 'http://0.0.0.0:5000/api/classifyGender'
data = {'Name':'Joel'}
r = requests.post(url=api_url, data=data)
print(r.status_code,r.text)