import enum
import threading
from typing import Callable, Self

from cachetools import LRUCache
import freetype
import numpy as np


__all__ = ["GlyphRenderer", "FontLanguage", "FontStore"]


type RenderTransform[U] = Callable[[np.ndarray], U]


class FontLanguage(str, enum.Enum):
    SIMPLIFIED_CHINESE = "zh_cn"
    TRADITIONAL_CHINESE = "zh_tw"
    JAPANESE = "jp"
    KOREAN = "ko"
    ENGLISH = "en"


class FontStore:
    def __init__(self, faces: dict[FontLanguage, freetype.Face]):
        self.faces = faces

    @classmethod
    def from_path(cls, font_file_map: dict[FontLanguage, str]) -> Self:
        faces = {
            language: freetype.Face(path)
            for language, path in font_file_map.items()
        }

        return cls(faces)

    def get_face(self, *, language: FontLanguage | None = None, char: str | None = None) -> freetype.Face | None:
        if language is None:
            for language, face in self.faces.items():
                if face.get_char_index(char):
                    language = language
                    break

            if language is None:
                return None

        return self.faces[language]


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
            return np.zeros(pixel_size or (1, 1), dtype=np.uint8)

        if weight:
            face.set_var_design_coords((weight,))
        
        if pixel_size:
            face.set_pixel_sizes(*pixel_size)

        with self.lock:
            face.load_char(_char)

        glyph = face.glyph
        bitmap = np.array(glyph.bitmap.buffer, dtype=np.uint8).reshape(glyph.bitmap.rows, glyph.bitmap.width)
        self.render_cache[key] = bitmap

        return bitmap
