#!/usr/bin/env python3
# =============================================================================
# preprocessing/generate_descriptions.py — Unified LLaVA description generator
# =============================================================================
#
# Generates text descriptions for all objects in a rendered image directory
# using LLaVA 1.5-7B.  Each object gets one description from its first
# rendered view (view 0).
#
# Usage:
#   python3 preprocessing/generate_descriptions.py \
#       --images_dir object_images/tless/ \
#       --output object_database/tless/descriptions_attributes.json
#
# The script is idempotent: existing entries in the output JSON are skipped.
# Use --overwrite to regenerate all descriptions.
# =============================================================================

import argparse
import json
import os
import sys

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration


def parse_args():
    p = argparse.ArgumentParser(description="Generate LLaVA descriptions for rendered CAD images")
    p.add_argument("--images_dir", required=True, help="Directory with rendered images (object_images/{dataset}/)")
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--overwrite", action="store_true", help="Regenerate all descriptions")
    p.add_argument("--prompt", default="Extract visual attributes of the object in the image: object type, brand name, color, material, and label text.",
                   help="LLaVA prompt")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Images captioned per forward pass. Every image still gets "
                        "its own caption; batching only improves GPU throughput.")
    return p.parse_args()


def _strip_assistant(text):
    text = text.strip()
    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    return text


def _encode_one(processor, image, prompt):
    """Encode a single image+prompt with the exact template the original
    one-at-a-time path used (proven correct)."""
    conversation = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]
    return processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def generate_captions(model, processor, images, prompt):
    """Caption a list of PIL images, returning one caption per image in order.

    Each image is encoded with the identical template the single-image path
    uses, then the encodings are stacked into one batch for a single forward
    pass. Every caption uses the same prompt and LLaVA-1.5 emits a fixed number
    of image tokens per image, so all sequences share the same length — the
    stack is exact (no padding) and greedy decoding yields the same text as
    captioning one image at a time. On CUDA OOM the batch is split and retried,
    so an over-large batch size degrades gracefully instead of failing.
    """
    if not images:
        return []
    encs = [_encode_one(processor, img, prompt) for img in images]
    batch = {k: torch.cat([e[k] for e in encs], dim=0) for k in encs[0]}
    for k in batch:
        batch[k] = batch[k].to(model.device)
    if "pixel_values" in batch:
        batch["pixel_values"] = batch["pixel_values"].to(torch.float16)

    try:
        with torch.no_grad():
            outputs = model.generate(**batch, max_new_tokens=100)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if len(images) == 1:
            raise
        mid = len(images) // 2
        return (generate_captions(model, processor, images[:mid], prompt)
                + generate_captions(model, processor, images[mid:], prompt))

    return [_strip_assistant(t) for t in processor.batch_decode(outputs, skip_special_tokens=True)]


def main():
    args = parse_args()

    # Load existing descriptions if present
    if os.path.exists(args.output) and not args.overwrite:
        with open(args.output, "r") as f:
            descriptions = json.load(f)
        print(f"Loaded {len(descriptions)} existing descriptions from {args.output}")
    else:
        descriptions = {}

    # Discover objects
    if not os.path.isdir(args.images_dir):
        print(f"ERROR: Images directory not found: {args.images_dir}")
        sys.exit(1)

    object_dirs = sorted([
        d for d in os.listdir(args.images_dir)
        if os.path.isdir(os.path.join(args.images_dir, d))
    ])

    # Filter to objects that need descriptions (partially described objects
    # are resumed — only missing images within them are captioned)
    if args.overwrite:
        pending = object_dirs
    else:
        pending = []
        for d in object_dirs:
            if d not in descriptions:
                pending.append(d)
            else:
                # Check if all images in this object have been described
                obj_dir = os.path.join(args.images_dir, d)
                image_files = sorted([
                    f for f in os.listdir(obj_dir)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
                    and not f.endswith("_bg.png")
                ])
                existing = descriptions[d].get("image_descriptions", {})
                if len(existing) < len(image_files):
                    pending.append(d)

    print(f"Total objects: {len(object_dirs)}, already described: {len(descriptions)}, pending: {len(pending)}")

    if not pending:
        print("All objects fully described. Use --overwrite to regenerate.")
        return

    # Load model
    print("Loading LLaVA 1.5-7B...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LlavaForConditionalGeneration.from_pretrained(
        "llava-hf/llava-1.5-7b-hf", torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
    print(f"Model loaded on {device}")

    # Generate descriptions for ALL views of each object
    # (matches original OSCAR behavior: one caption per rendered image)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    total_captions = 0

    for i, obj_id in enumerate(pending):
        obj_dir = os.path.join(args.images_dir, obj_id)

        image_files = sorted([
            f for f in os.listdir(obj_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
            and not f.endswith("_bg.png")
        ])

        if not image_files:
            print(f"  [{i+1}/{len(pending)}] SKIP {obj_id}: no images found")
            continue

        existing_data = descriptions.get(obj_id, {})
        image_descriptions = existing_data.get("image_descriptions", {})

        pending_images = [f for f in image_files if f not in image_descriptions or args.overwrite]
        if not pending_images:
            print(f"  [{i+1}/{len(pending)}] {obj_id}: all {len(image_files)} images already described")
            continue

        for k in range(0, len(pending_images), args.batch_size):
            batch_files = pending_images[k:k + args.batch_size]
            try:
                images = [Image.open(os.path.join(obj_dir, f)).convert("RGB") for f in batch_files]
                captions = generate_captions(model, processor, images, args.prompt)
                for filename, caption in zip(batch_files, captions):
                    image_descriptions[filename] = caption
                    total_captions += 1
                done = min(k + len(batch_files), len(pending_images))
                print(f"  [{i+1}/{len(pending)}] {obj_id} [{done}/{len(pending_images)}] {batch_files[-1]}: {captions[-1][:60]}...")
            except Exception as e:
                print(f"  [{i+1}/{len(pending)}] ERROR {obj_id} batch@{k}: {e}")
                continue

        descriptions[obj_id] = {"image_descriptions": image_descriptions}
        print(f"  [{i+1}/{len(pending)}] {obj_id}: {len(image_descriptions)} images done")

        # Save after every object (each takes ~2 min with 42 views,
        # so losing progress on interrupt is expensive)
        with open(args.output, "w") as f:
            json.dump(descriptions, f, indent=2)

    # Final save
    with open(args.output, "w") as f:
        json.dump(descriptions, f, indent=2)
    print(f"Done. {len(descriptions)} objects, {total_captions} new captions saved to {args.output}")


if __name__ == "__main__":
    main()
