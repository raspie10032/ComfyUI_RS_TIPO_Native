from .nodes import RSTIPONativeCompletion, RSTIPONativeExpand

NODE_CLASS_MAPPINGS = {
    "RSTIPONativeCompletion": RSTIPONativeCompletion,
    "RSTIPONativeExpand": RSTIPONativeExpand,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RSTIPONativeCompletion": "RS TIPO Native Completion (Experimental)",
    "RSTIPONativeExpand": "RS TIPO Native Expand (Experimental)",
}
