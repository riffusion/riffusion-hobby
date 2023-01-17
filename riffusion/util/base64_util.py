import base64
import io


def encode(buffer: io.BytesIO) -> str:
    """
    Encode the given buffer as base64.
    """
    return base64.encodebytes(buffer.getvalue()).decode("ascii")
