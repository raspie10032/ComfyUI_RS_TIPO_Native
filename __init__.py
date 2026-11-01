from .nodes import (RSTIPONativeCompletion, RSTIPONativeExpand,
                    RSTIPONativeControlledExpand, RSTIPONativePipeExpand, RSTIPOPrompt)

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "RSTIPOPrompt": RSTIPOPrompt,
    "RSTIPONativeCompletion": RSTIPONativeCompletion,
    "RSTIPONativeExpand": RSTIPONativeExpand,
    "RSTIPONativeControlledExpand": RSTIPONativeControlledExpand,
    "RSTIPONativePipeExpand": RSTIPONativePipeExpand,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RSTIPOPrompt": "RS TIPO Prompt (Experimental)",
    "RSTIPONativeCompletion": "RS TIPO Native Completion (Experimental)",
    "RSTIPONativeExpand": "RS TIPO Native Expand (Experimental)",
    "RSTIPONativeControlledExpand": "RS TIPO Native Controlled Expand (Experimental)",
    "RSTIPONativePipeExpand": "RS TIPO Native Pipe Expand (Experimental)",
}
