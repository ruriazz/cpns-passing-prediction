from marshmallow import fields, validate
from src.api.validators.base import BaseJsonSchema


class PredictionRequestSchema(BaseJsonSchema):
    no_peserta = fields.String(required=False)
    nama = fields.String(required=False)
    umur = fields.Integer(required=True, validate=validate.Range(min=18, max=60))
    nilai_ipk = fields.Float(required=True, validate=validate.Range(min=0, max=4.0))
    nilai_skd = fields.Float(required=True, validate=validate.Range(min=0))
    nilai_skb = fields.Float(required=True, validate=validate.Range(min=0))
