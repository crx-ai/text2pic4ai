import argparse
from functools import partial
from pathlib import Path

from PIL import Image
from datasets import load_dataset
import pyarrow as pa

from text2pic4ai.freetype import FontLanguage, GlyphRenderer
from text2pic4ai.freetype import FontStore
from text2pic4ai.processor import BitmapSentenceProcessor
from text2pic4ai.pyarrow_io import PyArrowBitmapSequenceSerializer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pixel-size", type=int, required=True)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--font-map", type=dict, default={})
    args = parser.parse_args()

    font_file_map = args.font_map or {
        FontLanguage.ENGLISH: "data/Noto_Sans/NotoSans-VariableFont_wdth,wght.ttf",
        FontLanguage.SIMPLIFIED_CHINESE: "data/Noto_Sans_SC/NotoSansSC-VariableFont_wght.ttf",
        FontLanguage.TRADITIONAL_CHINESE: "data/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf",
    }

    processor = BitmapSentenceProcessor(font_file_map=font_file_map, pixel_size=(args.pixel_size, args.pixel_size))
    dataset = load_dataset(args.dataset)

    for split in dataset:
        ds = dataset[split]
        
        for example in ds:
            bitmaps = processor(text=example[args.text_column], return_tensors="pt")
            Image.fromarray(bitmaps.pixel_values[0].numpy()).show()
            input("Next...")


if __name__ == "__main__":
    main()
