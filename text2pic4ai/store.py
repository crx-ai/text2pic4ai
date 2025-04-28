import enum
from typing import Self

import freetype


__all__ = ["FontLanguage", "FontStore"]


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
