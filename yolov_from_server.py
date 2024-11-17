from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="3BDEDCPYaZmmWDOEAaSN"
)

import time
start = time.perf_counter()
result = CLIENT.infer('test_input/photo_5316571971284232540_y.jpg', model_id="trafficsigndetection-vwdix/10")
end = time.perf_counter()
print(result)
print(end-start)