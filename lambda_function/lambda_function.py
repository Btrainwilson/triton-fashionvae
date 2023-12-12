import json
import boto3
from PIL import Image
import io
import time

def lambda_handler(event, context):
    # TODO implement
    if 'x' not in event:
        event['x'] = 0.5
    if 'y' not in event:
        event['y'] = 0.5

    payload = {
    "inputs": [
        {
            "name": "INPUT__0",
            "shape": [1, 2],
            "datatype": "FP32",
            "data": [event['x'], event['y']]
        }
        ]
    }

    runtime_sm_client = boto3.client("sagemaker-runtime")
    endpoint_name = "triton-fvae-pt-2023-12-11-16-24-51-Endpoint-20231211-162554"
    
    response = runtime_sm_client.invoke_endpoint(
    EndpointName="triton-fvae-pt-2023-12-11-16-24-51-Endpoint-20231211-162554",
    Body=json.dumps(payload),
    ContentType="application/json",
    Accept='accept',
    InferenceComponentName="triton-fvae-pt-2023-12-11-17-00-36-20231211-170152"
    )
    
    response = json.loads(response["Body"].read().decode("utf8"))
    image_data = response['outputs'][0]['data']  # Assuming this is a flat list of 28*28 elements

    # Manually reshape the data into a 28x28 array
    image_data_reshaped = [image_data[i:i+28] for i in range(0, len(image_data), 28)]

    # Create an image from the reshaped data
    image = Image.new('L', (28, 28))  # 'L' mode for grayscale
    for y, row in enumerate(image_data_reshaped):
        for x, value in enumerate(row):
            image.putpixel((x, y), int(value * 255))
    # Save image to a buffer
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)

    # Define S3 bucket and image path
    s3_bucket = "sagemaker-studio-340390718511-u03ftxnzm8"
    image_path = "fashion_mnist_imgs/triton-fvae-img-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime()) + '.png'

    # Upload to S3
    s3_client = boto3.client('s3')
    s3_client.upload_fileobj(buffer, s3_bucket, image_path)

    img_url = "https://sagemaker-studio-340390718511-u03ftxnzm8.s3.amazonaws.com/"
    
    return {
        'statusCode': 200,
        'headers': {
            'Access-Control-Allow-Origin': '*',
            "Access-Control-Allow-Credentials": True
        },
        'body': {
            'image_path': img_url + image_path,
        }
    }
