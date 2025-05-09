import argparse
from functools import partial
from pathlib import Path

from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor

from text2pic4ai import FontLanguage, BitmapSentenceProcessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pixel-size", type=int, required=True)
    parser.add_argument("--text-column", type=str, default="text")
    args = parser.parse_args()

    processor = BitmapSentenceProcessor(pixel_size=(args.pixel_size, args.pixel_size))
    dataset = load_dataset(args.dataset)

    for split in dataset:
        ds = dataset[split]
        
        for example in ds:
            bitmaps = processor(text=example[args.text_column], return_tensors="pt")
            Image.fromarray(bitmaps.pixel_values[0].numpy()).show()
            input("Next...")


if __name__ == "__main__":
    main()
