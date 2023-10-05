# Project Cenglass

## detection server 

test code for detection server

```bash

curl -X GET http://localhost:21560/about 
curl -X POST -F "image=@test.jpg" http://localhost:21560/detect
curl -X POST -F "image=@test.jpg" http://192.168.0.215:21560/detect

```