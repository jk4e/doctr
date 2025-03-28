from doctr.io import DocumentFile
from doctr.models import ocr_predictor

import torch
from doctr.models import ocr_predictor, db_resnet50, fast_base

# Load custom detection model
det_model = fast_base(pretrained=False, pretrained_backbone=False)
det_params = torch.load('fast_base_20250328-095304.pt', map_location="cpu")
det_model.load_state_dict(det_params)
predictor = ocr_predictor(det_arch=det_model, pretrained=True)

print(ocr_predictor)

# Image
img = DocumentFile.from_images("data/21NZdTfk6AL._AC_.jpg")
# img = DocumentFile.from_images("path/to/your/img.jpg")

# Analyze
result = predictor(img)

result.show()
