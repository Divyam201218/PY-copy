import torch
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import base64
import io
import numpy as np

# Device and model setup (outside function to avoid reloads)
device = torch.device("cpu")
model = InceptionResnetV1(pretrained='vggface2').eval().to(device).half()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.half()),  # Use half precision
    transforms.Normalize([0.5], [0.5])
])

def decode_base64_image(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    return img

def preprocess_image(img):
    return transform(img).unsqueeze(0).to(device)

def get_embedding_from_base64(base64_str):
    img = decode_base64_image(base64_str)
    tensor = preprocess_image(img)

    with torch.no_grad():
        embedding = model(tensor)

    embedding = embedding.cpu().numpy().flatten()
    embedding /= np.linalg.norm(embedding)  # L2 normalize
    return embedding.tolist()
