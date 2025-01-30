import tkinter as tk
from tkinter import filedialog
import cv2
import torch
import os
import config
from model import YOLOv3
from PIL import Image, ImageTk, ImageDraw
from utils import convert_to_bboxes, non_max_suppression

# Initialize model
model = YOLOv3(num_classes=config.NUM_CLASSES).to(config.DEVICE)

# Load checkpoint
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "YOLOv3_epoch_28.pth")
checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

class YOLOApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLOv3 Object Detector")
        
        # Create GUI elements
        self.button = tk.Button(root, text="Select Image", command=self.load_image)
        self.button.pack(pady=10)
        
        self.image_label = tk.Label(root)
        self.image_label.pack()

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        if file_path:
            self.process_and_display_image(file_path)

    def process_and_display_image(self, image_path):
        # Process image through YOLO pipeline
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (416, 416))
        image_tensor = torch.tensor(image / 255.0, dtype=torch.float32)
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(config.DEVICE)

        # Get predictions
        bboxes = self.get_predictions(image_tensor)

        # Draw bounding boxes and display
        self.display_results(image_tensor, bboxes)

    def get_predictions(self, image_tensor):
        scaled_anchors = torch.tensor(config.ANCHORS, device=config.DEVICE) / (
            1 / torch.tensor([13, 26, 52], device=config.DEVICE)
                .unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
        )
        
        with torch.no_grad():
            outputs = model(image_tensor)
            bboxes = [[] for _ in range(image_tensor.shape[0])]
            
            for i in range(3):
                batch_size, A, S, _, _ = outputs[i].shape
                anchor = scaled_anchors[i]
                boxes_scale_i = convert_to_bboxes(
                    outputs[i], anchor, S=S, is_preds=True
                )
                for idx, box in enumerate(boxes_scale_i):
                    bboxes[idx] += box

        return non_max_suppression(
            torch.tensor(bboxes[0], device=config.DEVICE), 
            iou_threshold=0.3, 
            threshold=0.4
        ).tolist()

    def display_results(self, image_tensor, bboxes):
        # Convert tensor to PIL Image
        image_np = (image_tensor[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
        pil_image = Image.fromarray(image_np)
        
        # Draw bounding boxes
        draw = ImageDraw.Draw(pil_image)
        for box in bboxes:
            x, y, w, h = box[2:6]
            x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
            x1, y1, x2, y2 = x1 * 416, y1 * 416, x2 * 416, y2 * 416
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            if len(box) > 5:  # If class information is available
                class_id = int(box[0])
                draw.text((x1, y1-20), str(config.CLASSES[class_id]), fill="red")

        # Display in GUI
        photo = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=photo)
        self.image_label.image = photo

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOApp(root)
    root.mainloop()