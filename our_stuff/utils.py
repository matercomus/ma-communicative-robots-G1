import numpy as np
from PIL import Image
import base64
import io


def numpy_to_base64(image_array, image_format="PNG"):
    """
    Convert a numpy array to a Base64-encoded string.

    Parameters:
        image_array (numpy.ndarray): The image array to convert.
        image_format (str): The desired image format (e.g., "PNG", "JPEG").

    Returns:
        str: Base64-encoded string of the image.
    """
    # Ensure the array is C-contiguous
    image_array = np.ascontiguousarray(image_array)

    # Save to in-memory buffer
    buffer = io.BytesIO()
    Image.fromarray(image_array).save(buffer, format=image_format)

    # Encode to Base64 and return as string
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
