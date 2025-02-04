from typing import Any, Dict
from marshmallow import Schema, fields, ValidationError


class BaseJsonSchema(Schema):
    """Base schema yang dapat digunakan untuk schema lain."""

    @classmethod
    def validate_request(cls, data: Dict[str, Any]) -> Any:
        """Validasi data menggunakan schema yang diturunkan."""
        try:
            return cls().load(data)
        except ValidationError as e:
            return {"errors": e.messages}
