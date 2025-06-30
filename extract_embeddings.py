import torchreid, cv2, numpy as np
import torchreid.data.transforms as T
import torch
from PIL import Image 

model = torchreid.models.build_model(name='osnet_x1_0', num_classes=1000, pretrained=True)
model.eval()
transform = T.build_transforms(height=256, width=128, is_train=False)[0]


def extract_reid_embedding(image):
    # Convert OpenCV (BGR) to RGB
    img = cv2.cvtColor(cv2.resize(image, (128, 256)), cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    img = Image.fromarray(img)

    # Apply transform
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        feat = model(img).cpu().numpy().flatten()

    return feat / np.linalg.norm(feat)
