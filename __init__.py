from .nodes import (RSTIPONativeCompletion, RSTIPONativeExpand,
                    RSTIPONativeControlledExpand, RSTIPONativePipeExpand)

NODE_CLASS_MAPPINGS = {
    "RSTIPONativeCompletion": RSTIPONativeCompletion,
    "RSTIPONativeExpand": RSTIPONativeExpand,
    "RSTIPONativeControlledExpand": RSTIPONativeControlledExpand,
    "RSTIPONativePipeExpand": RSTIPONativePipeExpand,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RSTIPONativeCompletion": "RS TIPO Native Completion (Experimental)",
    "RSTIPONativeExpand": "RS TIPO Native Expand (Experimental)",
    "RSTIPONativeControlledExpand": "RS TIPO Native Controlled Expand (Experimental)",
    "RSTIPONativePipeExpand": "RS TIPO Native Pipe Expand (Experimental)",
}
