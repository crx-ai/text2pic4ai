from functools import lru_cache
import threading
from typing import Self

from cachetools import LRUCache
import freetype
from freetype.ft_enums.ft_load_flags import FT_LOAD_DEFAULT
from freetype.ft_enums.ft_render_modes import FT_RENDER_MODE_NORMAL
import numpy as np

from text2pic4ai.store import FontLanguage, FontStore


__all__ = ["GlyphRenderer"]


class GlyphRenderer:
    def __init__(self, store: FontStore, *, cache_size: int = 500000):
        self.store = store
        self.render_cache = LRUCache(maxsize=cache_size)
        self.lock = threading.Lock()

    def render(
        self,
        string: str | None = None,
        *,
        language: FontLanguage | None = None,
        _char: str | None = None,
        pixel_size: tuple[int, int] = None,
        weight: int | None = None,
        limit: int | None = None,
    ) -> np.ndarray | None | list[np.ndarray]:
        if _char is None:
            if len(string) == 1:
                return self.render(language=language, _char=string, pixel_size=pixel_size, weight=weight)
            else:
                return [self.render(language=language, _char=c, pixel_size=pixel_size, weight=weight) for c in string[:limit]]
        
        try:
            key = (_char, pixel_size, weight)
            return self.render_cache[key]
        except KeyError:
            pass

        face = self.store.get_face(char=_char)

        if face is None:
            return None

        if weight:
            face.set_var_design_coords((weight,))
        
        if pixel_size:
            face.set_pixel_sizes(*pixel_size)

        with self.lock:
            face.load_char(_char)

        glyph = face.glyph
        bitmap = 255 - np.array(glyph.bitmap.buffer, dtype=np.uint8).reshape(glyph.bitmap.rows, glyph.bitmap.width)
        self.render_cache[key] = bitmap

        return bitmap
