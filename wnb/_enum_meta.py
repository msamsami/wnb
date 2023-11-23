from enum import EnumMeta, Enum
from typing import Any


class CaseInsensitiveEnumMeta(EnumMeta):
    """
    Enum metaclass to allow for interoperability with case-insensitive strings.
    """

    def __getitem__(cls, name: str) -> Any:
        return super(CaseInsensitiveEnumMeta, cls).__getitem__(name.upper())

    def __getattr__(cls, name: str) -> Enum:
        """Returns the enum member matching `name`.

        We use __getattr__ instead of descriptors or inserting into the enum
        class' __dict__ in order to support `name` and `value` being both
        properties for enum members (which live in the class' __dict__) and
        enum members themselves.

        :param str name: The name of the enum member to retrieve.
        :rtype: ~CaseInsensitiveEnumMeta
        :return: The enum member matching `name`.
        :raises AttributeError: If `name` is not a valid enum member.
        """
        try:
            return cls._member_map_[name.upper()]
        except KeyError as err:
            raise AttributeError(name) from err
