import argparse
from functools import partial
from pathlib import Path

from datasets import load_dataset
import pyarrow as pa

from text2pic4ai.freetype import FontLanguage, GlyphRenderer
from text2pic4ai.freetype import FontStore
from text2pic4ai.processor import BitmapSentenceProcessor
from text2pic4ai.pyarrow_io import PyArrowBitmapSequenceSerializer


global_font_store: FontStore | None = None
global_renderer: GlyphRenderer | None = None


def get_renderer(font_file_map: dict[FontLanguage, str]) -> GlyphRenderer:
    global global_font_store, global_renderer

    if global_font_store is None or global_renderer is None:
        global_font_store = FontStore.from_path(font_file_map)
        global_renderer = GlyphRenderer(global_font_store)

    return global_renderer


def render_text(text_column: str, pixel_size: int, font_file_map: dict[FontLanguage, str], limit: int, example: dict):
    renderer = get_renderer(font_file_map)
    text = example[text_column]
    bitmaps = renderer.render(text, pixel_size=(pixel_size, pixel_size), limit=limit)    
    serializer = PyArrowBitmapSequenceSerializer()

    for k, v in serializer.serialize(bitmaps).items():
        example[k] = v

    return example


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--pixel-size", type=int, required=True)
    parser.add_argument("--output-folder", "-o", type=Path, required=True)
    parser.add_argument("--text-column", type=str, default="text")
    parser.add_argument("--num-jobs", "-j", type=int, default=8)
    parser.add_argument("--font-map", type=dict, default={})
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    font_file_map = args.font_map or {
        FontLanguage.ENGLISH: "data/Noto_Sans/NotoSans-VariableFont_wdth,wght.ttf",
        FontLanguage.SIMPLIFIED_CHINESE: "data/Noto_Sans_SC/NotoSansSC-VariableFont_wght.ttf",
        FontLanguage.TRADITIONAL_CHINESE: "data/Noto_Sans_TC/NotoSansTC-VariableFont_wght.ttf",
    }

    dataset = load_dataset(args.dataset)

    for split in dataset:
        ds = dataset[split]
        ds = ds.map(
            partial(render_text, args.text_column, args.pixel_size, font_file_map, args.limit),
            num_proc=args.num_jobs,
            writer_batch_size=1000,
        )
        ds.features.update(PyArrowBitmapSequenceSerializer().get_features())
        dataset[split] = ds

    dataset.save_to_disk(args.output_folder)


if __name__ == "__main__":
    main()
