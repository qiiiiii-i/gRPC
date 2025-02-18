import io
from collections import Counter
import grpc
from concurrent import futures
import sei_pb2
import sei_pb2_grpc
import os
import torch
import numpy as np
import time
from aaresnet1dPiles import *
import sys
import base64
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from collections import Counter

def sun(encrypted_weights):
    key = b"\xeb\x9c)k!24'\xdfY@r\x1e\xc1\xee\xca\x163\xabFs\xae\xc3P\x9c\xdc\xbe/\x9b\xe94\x88"
    iv = b'7oZ\xc2$\x91\x90Z\xafq\xc9W\xc1\xb0\xe0\xba'

    cipher = Cipher(algorithms.AES(key), modes.CFB(iv))
    decryptor = cipher.decryptor()

    decoded = decryptor.update(encrypted_weights) + decryptor.finalize()

    try:
        # Base64 解码权重
        weights_bytes = base64.b64decode(decoded)
        print(f" 解码成功")
    except Exception as e:
        print(f"解码失败: {e}")
        return None

    buffer = io.BytesIO(weights_bytes)
    state_dict = torch.load(buffer)

    return state_dict
