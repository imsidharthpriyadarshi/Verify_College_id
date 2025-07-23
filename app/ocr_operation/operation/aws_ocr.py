import boto3
import time

textract = boto3.client('textract')

def extract(image_path):
    
    with open(image_path, 'rb') as document:
        image_bytes = document.read()

    response = textract.detect_document_text(
        Document={'Bytes': image_bytes}
    )

    print('Detected Text:')
    for item in response['Blocks']:
        if item['BlockType'] == 'LINE':
            print(f"Detected Line: {item['Text']}")

