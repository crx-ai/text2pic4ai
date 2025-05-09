from typing import Protocol

import numpy as np
import pyarrow as pa
import datasets


class PyArrowSerializer[T, U: pa.DataType](Protocol):
    def serialize(self, data: T) -> dict[str, U]: ...    
    def deserialize(self, columns: dict[str, U]) -> T: ...
    def get_features(self) -> datasets.Features: ...


class PyArrowBitmapSequenceSerializer:
    def __init__(self, *, shape_column: str = "shapes", data_column: str = "bytes"):
        self.shape_column = shape_column
        self.data_column = data_column
    
    def get_features(self) -> datasets.Features:
        return datasets.Features({
            self.shape_column: datasets.Sequence(datasets.Sequence(datasets.Value(dtype="int32", id=None))),
            self.data_column: datasets.Sequence(datasets.Value(dtype="binary", id=None)),
        })

    def serialize(self, data: list[np.ndarray]) -> dict[str, pa.Array]:
        assert data[0].dtype == np.uint8, "Bitmap must be uint8"
        shapes = [bitmap.shape for bitmap in data]

        return {
            self.shape_column: shapes,
            self.data_column: [bitmap.tobytes() for bitmap in data],
        }
    
    def deserialize(self, columns: dict[str, pa.Array]) -> list[np.ndarray]:
        return [
            np.frombuffer(bytes, dtype=np.uint8).reshape(shape) for shape, bytes in zip(columns[self.shape_column], columns[self.data_column])
        ]
