from dataclasses import dataclass, asdict
from typing import List


@dataclass
class NguoiDung:

    id: int
    name: str
    age: str
    home: str
    class_name: str
    major: str
    embedding: List[float]
    embedding_dimension: int = 512
    image_count: int = 0

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(data):
        return NguoiDung(
            id=int(data["id"]),
            name=data.get("name", ""),
            age=data.get("age", ""),
            home=data.get("home", ""),
            class_name=data.get("class_name", data.get("class", "")),
            major=data.get("major", ""),
            embedding=list(data["embedding"]),
            embedding_dimension=int(data.get("embedding_dimension", 512)),
            image_count=int(data.get("image_count", 0))
        )