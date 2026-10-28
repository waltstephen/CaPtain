"""Concise public argument constraints; authoritative validation stays in JSON Schema."""
from collections.abc import Mapping
from copy import deepcopy

_RULES = ('enum','const','minimum','maximum','exclusiveMinimum','exclusiveMaximum',
          'multipleOf','minLength','maxLength','pattern','minItems','maxItems','uniqueItems',
          'minProperties','maxProperties','oneOf','anyOf','allOf','not','dependentRequired')


def _field_constraints(field):
    selected = {key:deepcopy(field[key]) for key in _RULES if key in field}
    # Scalar bounds on array elements are executable constraints too (for
    # example explicit angles); avoid expanding nested object documentation.
    if isinstance(field.get('items'), Mapping):
        items = _field_constraints(field['items'])
        if items:
            selected['items'] = items
    return selected


def public_argument_constraints(schema):
    arguments = {}
    nested = []
    for name, field in schema.get('properties', {}).items():
        if not isinstance(field, Mapping):
            continue
        selected = _field_constraints(field)
        if selected:
            arguments[name] = selected
        if 'properties' in field or field.get('type') == 'object' or 'object' in (field.get('type') or []):
            nested.append(name)
    result = {'arguments':arguments}
    for key in ('required','oneOf','anyOf','allOf','not','dependentRequired'):
        if key in schema:
            result[key] = deepcopy(schema[key])
    if nested:
        result['nested_object_arguments'] = nested
        result['detail_access'] = 'Use inspect_capability for full nested schemas; this summary does not replace validation.'
    return result


def inferred_literal_type(schema):
    """Infer primitive documentation types for schemas expressed by allowed values."""
    if 'type' in schema:
        return schema['type']
    values = schema.get('enum') if isinstance(schema.get('enum'), list) else None
    if values is None and 'const' in schema:
        values = [schema['const']]
    if not values:
        return None
    def kind(value):
        if value is None: return 'null'
        if isinstance(value, bool): return 'boolean'
        if isinstance(value, str): return 'string'
        if isinstance(value, int): return 'integer'
        if isinstance(value, float): return 'number'
        if isinstance(value, list): return 'array'
        return 'object'
    kinds = list(dict.fromkeys(kind(value) for value in values))
    return kinds[0] if len(kinds)==1 else kinds
