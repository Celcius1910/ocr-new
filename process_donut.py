from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import cv2
from ultralytics import YOLO

droidcam_url = "http://192.168.1.3:4747/video"

# Load processor and model from Hugging Face
model_id = "./donut-ktp-v3"
processor = DonutProcessor.from_pretrained(model_id)
model = VisionEncoderDecoderModel.from_pretrained(model_id)

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)


model_yolo = YOLO("best.pt")  # or "best.pt" for your card-trained model

# Open video stream

cap = cv2.imread("C:\\Users\\fnugr\\Downloads\\20251008_144444.jpg")

results = model_yolo(cap, verbose=False)

for result in results:
    boxes = result.boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        cropped = cap[y1:y2, x1:x2]

        cv2.imwrite("ktp_croped.jpg", cropped)

img = cv2.imread("ktp_croped.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
image = Image.fromarray(img)

# Prepare input
pixel_values = processor(image, return_tensors="pt").pixel_values
pixel_values = pixel_values.to(device)

# Generate prediction
task_prompt = "<s_ktp>"   # model-specific prefix (you can test empty prompt if unsure)
outputs = model.generate(pixel_values, max_length=512, num_beams=4)

# Decode
decoded = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print(decoded)
