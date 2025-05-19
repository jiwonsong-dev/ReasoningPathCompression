from importlib.metadata import version
import warnings
import transformers

from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES
from transformers.models.qwen2.modeling_qwen2 import QWEN2_ATTENTION_CLASSES

from rpc.llama_custom import LlamaRPCAttention
from rpc.qwen2_custom import Qwen2RPCAttention


def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    version_list = ['4.45']
    warning_flag = True
    for x in version_list:
        if x in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")


def enable_rpc():
    check_version()

    LLAMA_ATTENTION_CLASSES['flash_attention_2'] = LlamaRPCAttention
    QWEN2_ATTENTION_CLASSES['flash_attention_2'] = Qwen2RPCAttention