from google.cloud import vision
from google.cloud.vision import types
import io

client = vision.ImageAnnotatorClient()

def extract(image_path):
    
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = types.Image(content=content)

    response = client.text_detection(image=image)
    
    texts = response.text_annotations
    
    if texts:
        print('Detected text:')
        return texts
        #for text in texts:
            #print(text.description)        
    else:
        print('No text detected')
       

    if response.error.message:
        raise Exception(f"Error in API request: {response.error.message}")
    return ""

#if __name__ == '__main__':
#    image_path = 'path_to_your_image.jpg'  
#    detect_text_from_image(image_path)
