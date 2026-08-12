"""Machine-readable verdict schema v1 + a dependency-free structural validator.

The schema file (verdict.schema.json) is the frozen wire contract (docs/plan.md
§2.2); verdict.validate() enforces the cross-field invariants a JSON Schema can't
express (PASS ⇒ layers_run == [l1,l2], etc.). Both run on every verdict in tests.

The validator supports the subset of JSON Schema used by our schema files:
type (string or list), properties, required, items, enum, additionalProperties
(bool), minimum/maximum for numbers, and $ref into #/$defs/.
"""

from __future__ import annotations

import json
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent / "verdict.schema.json"

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))


def _type_ok(value, tname: str) -> bool:
    if tname == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if tname == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    py = _TYPES.get(tname)
    if py is dict or py is list or py is str or py is type(None):
        return isinstance(value, py)
    if py is bool:
        return isinstance(value, bool)
    return True


def check_schema(instance, schema: dict, path: str = "$", root: dict | None = None) -> list[str]:
    root = root if root is not None else schema
    ref = schema.get("$ref")
    if isinstance(ref, str) and ref.startswith("#/"):
        target = root
        for part in ref[2:].split("/"):
            target = target[part]
        return check_schema(instance, target, path, root)

    errs: list[str] = []
    types = schema.get("type")
    if types is not None:
        tlist = types if isinstance(types, list) else [types]
        if not any(_type_ok(instance, t) for t in tlist):
            errs.append(f"{path}: type {type(instance).__name__} not in {tlist}")
            return errs
        if instance is None and "null" in tlist:
            return errs
    if "enum" in schema and instance not in schema["enum"]:
        errs.append(f"{path}: {instance!r} not in enum {schema['enum']}")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            errs.append(f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            errs.append(f"{path}: {instance} > maximum {schema['maximum']}")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errs.append(f"{path}: missing required key {key!r}")
        props = schema.get("properties", {})
        for key, sub in props.items():
            if key in instance:
                errs.extend(check_schema(instance[key], sub, f"{path}.{key}", root))
        if schema.get("additionalProperties") is False:
            extra = set(instance) - set(props)
            if extra:
                errs.append(f"{path}: unexpected keys {sorted(extra)}")
    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            errs.extend(check_schema(item, schema["items"], f"{path}[{i}]", root))
    return errs


def validate_verdict_file(verdict: dict) -> list[str]:
    """Schema conformance + cross-field invariants; empty list = valid."""
    from .verdict import validate as validate_invariants
    return check_schema(verdict, load_schema()) + validate_invariants(verdict)
