"""Device management (CPU, GPU, etc.)."""


class Device:
    """Compute device identifier."""

    __slots__ = ("type",)

    def __init__(self, device_type: str) -> None:
        self.type = device_type

    @classmethod
    def cpu(cls) -> "Device":
        return cls("cpu")

    @classmethod
    def cuda(cls, index: int = 0) -> "Device":
        return cls(f"cuda:{index}")

    def __repr__(self) -> str:
        return f"Device(type={self.type!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Device) and self.type == other.type

    def __hash__(self) -> int:
        return hash(self.type)
