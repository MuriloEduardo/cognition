import json
from typing import Any


def wrap_for_template(value: Any) -> Any:
    """Wrap dicts and lists in AttrView so templates can access them cleanly."""
    if isinstance(value, (dict, list)):
        return AttrView(value)
    return value


class AttrView:
    """
    Thin wrapper that makes dict/list values accessible as attributes in
    Jinja2 templates and stringifies them to JSON when interpolated.

    Usage in templates:
        {{ flow.system_prompt }}         → string value (or empty string)
        {{ flow.properties }}            → JSON array string
        {%- if flow.system_prompt %}     → truthy check
    """

    def __init__(self, data: Any) -> None:
        self._data = data

    def __repr__(self) -> str:
        return self._stringify()

    def __str__(self) -> str:
        return self._stringify()

    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(key)
        if isinstance(self._data, dict):
            return wrap_for_template(self._data.get(key))
        return getattr(self._data, key)

    def __bool__(self) -> bool:
        if self._data is None:
            return False
        if isinstance(self._data, (str, dict, list)):
            return bool(self._data)
        return True

    def _stringify(self) -> str:
        if self._data is None:
            return ""
        try:
            return json.dumps(self._data, ensure_ascii=False, default=repr)
        except TypeError:
            return repr(self._data)
