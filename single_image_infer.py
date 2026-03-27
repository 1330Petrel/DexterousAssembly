import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from torchvision.ops import box_convert

from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

"""
Hyper parameters
"""
root = str(Path(__file__).parent)
SAM2_CHECKPOINT = f"{root}/checkpoints/sam2.1_hiera_large.pt"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GROUNDING_DINO_CONFIG = f"{root}/grounding_dino/groundingdino/config/GroundingDINO_SwinB_cfg.py"
GROUNDING_DINO_CHECKPOINT = f"{root}/gdino_checkpoints/groundingdino_swinb_cogcoor.pth"

BOX_THRESHOLD = 0.315
TEXT_THRESHOLD = 0.25
DEBUG = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_prompt(prompt: str) -> str:
    prompt = prompt.strip().lower()
    if not prompt.endswith("."):
        prompt += "."
    return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input image (e.g. /tmp/input.jpg)")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for object detection (e.g. 'hammer')")
    parser.add_argument("--out", type=str, required=True, help="Path to output mask image (e.g. /tmp/mask.png)")
    args = parser.parse_args()

    img_path = Path(args.img)
    if not img_path.exists():
        print(f"[ERROR] Input image not found: {img_path}")
        sys.exit(1)

    text_prompt = parse_prompt(args.prompt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Device: {DEVICE}")

    # 1. build SAM2 image predictor
    sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    sam2_predictor = SAM2ImagePredictor(sam2_model)
    
    grounding_model = load_model(
        model_config_path=GROUNDING_DINO_CONFIG,
        model_checkpoint_path=GROUNDING_DINO_CHECKPOINT,
        device=DEVICE,
    )

    print(f"[INFO] Start inference: {img_path.name} -> {text_prompt}")

    # 2. setup the input image for SAM 2 and Grounding DINO
    image_source, image = load_image(str(img_path))
    sam2_predictor.set_image(image_source)
    h, w, _ = image_source.shape

    # 3. Run Grounding DINO to get bounding boxes
    boxes, confidences, class_names = predict(
        model=grounding_model,
        image=image,
        caption=text_prompt,
        box_threshold=BOX_THRESHOLD,
        text_threshold=TEXT_THRESHOLD,
        device=DEVICE,
    )

    if boxes.numel() == 0:
        print(f"[WARN] No detection, saving empty mask: {out_path.name}")
        empty_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.imwrite(str(out_path), empty_mask)
        sys.exit(0)

    top_idx = int(torch.argmax(confidences).item())
    boxes = boxes[top_idx : top_idx + 1]
    confidence = confidences[top_idx].item()
    print(f"[INFO] Box confidence {confidence:.4f}")

    boxes = boxes * torch.tensor([w, h, w, h], device=boxes.device)
    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").cpu().numpy()

    # 4. Run SAM 2 to get masks
    if DEVICE == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    with torch.autocast(device_type=DEVICE, dtype=torch.bfloat16):
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )
    print(f"[INFO] SAM 2 score: {scores[0]:.4f}")

    if masks.ndim == 4:
        masks = masks.squeeze(1)

    masks = masks.astype(bool)
    merged_mask = np.any(masks, axis=0).astype(np.uint8) * 255

    # 5. Save the merged mask
    cv2.imwrite(str(out_path), merged_mask)
    print(f"[OK] Saved mask: {out_path}")

    if DEBUG:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[ERROR] Failed to read image for debug visualization: {img_path}")
            sys.exit(1)

        labels = [f"{class_names[top_idx]}: {confidence:.2f}"]
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks.astype(bool),
            class_id=np.array([top_idx]),
        )

        box_annotator = sv.BoxAnnotator()
        annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

        label_annotator = sv.LabelAnnotator()
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        cv2.imwrite(str(out_path.parent / "debug.png"), annotated_frame)
        print(f"[DEBUG] Saved debug visualization: {out_path.parent / 'debug.png'}")

if __name__ == "__main__":
    main()

