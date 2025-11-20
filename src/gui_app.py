import gradio as gr
import cv2
import numpy as np
import easyocr
import json
from preprocess_receipt import preprocess_receipt

# Load OCR model once (GPU if available)
reader = easyocr.Reader(['en'], gpu=True)


def draw_boxes(image, results, font_scale=1.3, text_thickness=3, box_thickness=2):
    annotated = image.copy()

    for (bbox, text, prob) in results:
        pts = np.array(bbox, dtype=np.int32)

        x_min = np.min(pts[:, 0])
        y_min = np.min(pts[:, 1])
        x_max = np.max(pts[:, 0])
        y_max = np.max(pts[:, 1])

        # Box
        cv2.rectangle(
            annotated, (x_min, y_min), (x_max, y_max),
            (0, 255, 0), box_thickness, lineType=cv2.LINE_AA
        )

        label = f"{text}"

        # Text size
        (tw, th), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness
        )

        # Background rectangle
        bg_x1 = x_min
        bg_y1 = y_min - th - 12
        bg_x2 = x_min + tw + 20
        bg_y2 = y_min

        cv2.rectangle(
            annotated,
            (bg_x1, bg_y1),
            (bg_x2, bg_y2),
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

        # Outline text (black)
        cv2.putText(
            annotated, label,
            (x_min + 10, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            text_thickness + 2,
            cv2.LINE_AA
        )

        # Foreground text (green)
        cv2.putText(
            annotated, label,
            (x_min + 10, y_min - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 255, 0),
            text_thickness,
            cv2.LINE_AA
        )

    return annotated


def process_receipt(uploaded_img):
    if uploaded_img is None:
        return None, None, None, None

    # Convert from RGB (Gradio) to BGR (OpenCV)
    cv_img = cv2.cvtColor(uploaded_img, cv2.COLOR_RGB2BGR)

    # Preprocess
    original, preprocessed = preprocess_receipt(cv_img)

    # OCR
    results = reader.readtext(preprocessed)

    # Annotate
    annotated = draw_boxes(preprocessed.copy(), results)

    # Extract text
    text_only = "\n".join([res[1] for res in results])

    # JSON output
    json_output = json.dumps(
        [{"text": res[1], "confidence": float(res[2])} for res in results],
        indent=2
    )

    return (
        cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        text_only,
        json_output
    )


# ------------------------- GUI --------------------------

with gr.Blocks(title="Receipt OCR App") as demo:

    gr.Markdown("## 🧾 Receipt OCR — Powered by EasyOCR")

    with gr.Row():
        with gr.Column(scale=0.5):
            input_img = gr.Image(type="numpy", label="Upload Receipt")
            run_btn = gr.Button("Run OCR")

        with gr.Column(scale=0.5):
            pre_out = gr.Image(label="Preprocessed Image")
            ann_out = gr.Image(label="Annotated OCR Output")

    text_output = gr.Textbox(label="Extracted Text", lines=10)
    json_output = gr.Textbox(label="JSON Output", lines=10)

    run_btn.click(
        process_receipt,
        inputs=[input_img],
        outputs=[pre_out, ann_out, text_output, json_output]
    )

demo.launch()
