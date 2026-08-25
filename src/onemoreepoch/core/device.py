# Identifies a compute device (cpu, cuda:N) by type string
class Device:
    __slots__ = ("type",)

    # Stores the device type string
    def __init__(self, device_type: str) -> None:
        self.type = device_type

    # Returns a CPU device
    @classmethod
    def cpu(cls) -> "Device":
        return cls("cpu")

    # Returns a CUDA device with the given index
    @classmethod
    def cuda(cls, index: int = 0) -> "Device":
        return cls(f"cuda:{index}")

    # Returns a debug string representation
    def __repr__(self) -> str:
        return f"Device(type={self.type!r})"

    # Compares devices by type
    def __eq__(self, other: object) -> bool:
        return isinstance(other, Device) and self.type == other.type

    # Hashes by type so Devices can be used as dict keys / in sets
    def __hash__(self) -> int:
        return hash(self.type)
