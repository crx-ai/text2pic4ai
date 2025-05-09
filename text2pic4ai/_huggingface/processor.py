import importlib.resources as pkg_resources
from typing import TypedDict

import numpy as np
import torch
from transformers import ProcessorMixin, BatchEncoding
from transformers.processing_utils import ProcessingKwargs, CommonKwargs, Unpack
from transformers.tokenization_utils_base import TextInput

from .._pkg_data import get_pkg_data_path
from .._freetype import FontStore, GlyphRenderer, FontLanguage


class BitmapSentenceProcessingKwargs(ProcessingKwargs, total=False):
    class Common(CommonKwargs, total=False):
        pixel_size: tuple[int, int]

    _defaults = {
        "text_kwargs": {
            "padding": True,
            "stride": 0,
            "verbose": False,
            "padding_side": "right",
            "truncation": False,
            "max_length": 2048,
            "return_tensors": "pt",
            "return_attention_mask": True,
        },
        "images_kwargs": {},
    }
    
    common_kwargs: Common


def _get_default_font_file_map() -> dict[FontLanguage, str]:
    base_path = get_pkg_data_path()

    noto_sans_path = base_path / "Noto_Sans"
    noto_sans_sc_path = base_path / "Noto_Sans_SC"
    noto_sans_tc_path = base_path / "Noto_Sans_TC"

    return {
        FontLanguage.ENGLISH: str(noto_sans_path / "NotoSans-VariableFont_wdth,wght.ttf"),
        FontLanguage.SIMPLIFIED_CHINESE: str(noto_sans_sc_path / "NotoSansSC-VariableFont_wght.ttf"),
        FontLanguage.TRADITIONAL_CHINESE: str(noto_sans_tc_path / "NotoSansTC-VariableFont_wght.ttf"),
    }


class BitmapSentenceProcessor(ProcessorMixin):
    attributes = []
    valid_kwargs = ["pixel_size"]
    optional_attributes = ["pixel_size", "font_file_map", "font_weight"]
    
    def __init__(self, *args, **kwargs):        
        super().__init__(*args, **kwargs)
        self.font_store = FontStore.from_path(self.font_file_map or _get_default_font_file_map())
        self.renderer = GlyphRenderer(self.font_store)
        self.pixel_size = self.pixel_size or (14, 14)
        self.font_weight = self.font_weight or 500
    
    def merge_bitmaps(self, bitmaps: list[np.ndarray]) -> np.ndarray:
        resized_bitmaps = []

        for bitmap in bitmaps:
            current_height, current_width = bitmap.shape[:2]
            target_height, target_width = self.pixel_size

            pad_height = max(0, (target_height - current_height) // 2)
            pad_width = max(0, (target_width - current_width) // 2)

            new_bitmap = np.zeros((target_height, target_width), dtype=bitmap.dtype)
            start_y = pad_height
            start_x = pad_width
            bitmap = bitmap[:target_height, :target_width]

            new_bitmap[start_y:start_y + current_height, start_x:start_x + current_width] = bitmap
            resized_bitmaps.append(new_bitmap)

        return np.hstack(resized_bitmaps)

    def __call__(
        self,
        images=None,
        text: TextInput | list[TextInput] = None,
        audio=None,
        videos=None,
        **kwargs: Unpack[BitmapSentenceProcessingKwargs],
    ) -> BatchEncoding:
        if isinstance(text, str):
            text = [text]
        
        kwargs = self._merge_kwargs(
            BitmapSentenceProcessingKwargs,
            {},
            **kwargs
        )
        kwargs["text_kwargs"].setdefault("pixel_size", self.pixel_size)
        bitmaps_list = []

        for string in text:
            bitmaps = self.renderer.render(
                string,
                pixel_size=kwargs["text_kwargs"]["pixel_size"],
                weight=self.font_weight,
                limit=kwargs["text_kwargs"]["max_length"] if kwargs["text_kwargs"]["truncation"] else None
            )
            bitmaps_list.append(bitmaps)
        
        if kwargs["text_kwargs"]["padding"] == "max_length":
            attention_mask = [([1] * len(bitmaps)) + ([0] * (kwargs["text_kwargs"]["max_length"] - len(bitmaps))) for bitmaps in bitmaps_list]
            bitmaps_list = [bitmaps + [np.zeros(self.pixel_size, dtype=np.uint8)] * (kwargs["text_kwargs"]["max_length"] - len(bitmaps)) for bitmaps in bitmaps_list]
        elif kwargs["text_kwargs"]["padding"] or kwargs["text_kwargs"]["padding"] == "longest":
            attention_mask = [([1] * len(bitmaps)) + ([0] * (max(len(bitmaps) for bitmaps in bitmaps_list) - len(bitmaps))) for bitmaps in bitmaps_list]
            bitmaps_list = [bitmaps + [np.zeros(self.pixel_size, dtype=np.uint8)] * (max(len(bitmaps) for bitmaps in bitmaps_list) - len(bitmaps)) for bitmaps in bitmaps_list]
        else:
            attention_mask = None
        
        bitmaps_list = [self.merge_bitmaps(bitmaps) for bitmaps in bitmaps_list]
        
        if kwargs["text_kwargs"]["return_tensors"] == "pt":
            bitmaps_list = torch.stack([torch.from_numpy(x) for x in bitmaps_list])
            
            if attention_mask is not None:
                attention_mask = torch.stack([torch.tensor(attention_mask) for attention_mask in attention_mask])
        
        ret_dict = {"pixel_values": bitmaps_list}

        if kwargs["text_kwargs"]["return_attention_mask"]:
            ret_dict["attention_mask"] = attention_mask

        return BatchEncoding(ret_dict)
