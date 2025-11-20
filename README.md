🧾 Receipt OCR App
AI-powered app for automatic receipt text extraction using EasyOCR, OpenCV, and Gradio.
________________________________________
📘 Overview
This project demonstrates a complete Computer Vision + OCR pipeline, including:
•	Image preprocessing (contrast enhancement, deskewing, resizing)
•	OCR text extraction using EasyOCR
•	Annotation overlay (bounding boxes + readable labels)
•	JSON text export
•	Full interactive Gradio GUI
________________________________________
🚀 Features
🔧 Image Preprocessing
•	CLAHE contrast enhancement
•	Automatic deskew (Hough Transform)
•	Light denoising
•	Resize with preserved aspect ratio
•	Conversion to 3-channel BGR for OCR
🧠 OCR (EasyOCR)
•	Reads printed receipts
•	Extracts text + confidence
•	Robust to:
✔ shadows
✔ uneven lighting
✔ curved / warped receipts
✔ long receipts
🖼 Annotated Visualization
•	YOLO-style bounding boxes
•	White background behind text
•	High readability labels
•	Clean and professional output
💻 GUI (Gradio)
•	Upload any receipt (PNG/JPG)
•	View preprocessed image
•	See OCR bounding boxes
•	Extract raw text
•	Get JSON output
________________________________________
📂 Project Structure
receipt_ocr/
│
├── results/ 
│   ├── receipt1.png
│   └── receipt2.png
│
└── src/
    ├── gui_app.py
    └── preprocess_receipt.py
________________________________________
▶️ Steps to Run
1. Run the App
python src/gui_app.py
2. Open in Browser
👉 http://127.0.0.1:7860/
________________________________________
🔄 Preprocessing Pipeline
The preprocessing stage dramatically improves OCR performance:
Input Image →
Grayscale →
CLAHE →
Deskew →
Gaussian Blur →
Resize →
3-Channel Conversion →
EasyOCR
________________________________________
📸 Screenshots
⭐ Example 1 — Printed Receipt (Clean)
![Example1](/ results/receipt1.png)
⭐ Example 2 — Real-World Long Receipt (Complex)
![Example2](/ results/receipt2.png)
________________________________________
📝 Output Formats
🔤 Extracted Text (Raw)
Plain, readable text — ideal for downstream parsing.
🔧 JSON Output (example)
[
  {
    "text": "SALE",
    "confidence": 0.98
  },
  {
    "text": "TOTAL: $22.49",
    "confidence": 0.93
  }
]
________________________________________
🛠 Technologies Used
•	Python 3.10+
•	OpenCV (preprocessing)
•	EasyOCR (deep learning OCR model)
•	Gradio (UI)
•	NumPy
•	PyTorch (backend for EasyOCR)
________________________________________
🚧 Future Improvements
•	Table / line-item extraction
•	Automatic total / date detection via regex
•	Confidence thresholding
•	Web deployment (HuggingFace Spaces)
•	Handwritten OCR support (TrOCR)
________________________________________
👩‍💻 Author
Leila Bahman
Machine Vision & AI Specialist

