from preprocess import preprocess_image, image_to_tensor
import torch
from PIL import Image
from torchvision import transforms
from model import CRNN
from utility import ctc_decode, levenshtein_distance
from data_utility import preprocess_data
import pandas as pd
from model import CRNN  # Example import
import config
import json
# import itchat
# from itchat.content import PICTURE

# ------------------------------------------------------------


# Single image prediction
def predict_image(image_path, int_to_char, lexicon, num_classes, device='cpu'):
    # Define same preprocessing as used in training
    model = CRNN(num_classes=num_classes).to(device)   # initialize same architecture as before
    model.load_state_dict(torch.load("./crnn_model4.pth", map_location=torch.device('cpu')))
    model.eval()

    image, _ = preprocess_image(image_path, '')
    if image is None:
        return "Image load failed."
    
    img_tensor = image_to_tensor(image)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    print(img_tensor.shape) 
    # Load image and preprocess
    
    with torch.no_grad():
        outputs = model(img_tensor)  # (time_steps, batch, num_classes)
        decoded_text = ctc_decode(outputs, int_to_char, lexicon=lexicon)[0] # get single prediction
        print(decoded_text)
    return decoded_text

# itchat.auto_login(hotReload=True)
# itchat.run()
def main():
    device = 'cpu'

    with open("parameters.json", "r") as f:
        param = json.load(f)

    char_to_int = {k: int(v) for k, v in param["CHAR_TO_INT"].items()}
    int_to_char = {int(k): v for k, v in param["INT_TO_CHAR"].items()}
    lexicon = set(param["LEXICON"])
    num_classes = param["NUM_OF_CLASSES"]
    print(num_classes)
    decode = predict_image('./test.png', int_to_char, lexicon, num_classes)
    print(decode)

if __name__ == "__main__":
    main()

