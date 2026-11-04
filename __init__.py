from .nodes import (RSTIPONativeCompletion, RSTIPONativeExpand,
                    RSTIPONativeControlledExpand, RSTIPONativePipeExpand, RSTIPOPrompt)

WEB_DIRECTORY = "./web"
__version__ = "0.2.9"

# Only RSTIPOPrompt is offered for new workflows. Deprecated classes remain
# registered so existing graphs and research fixtures keep their original behavior.
NODE_CLASS_MAPPINGS = {
    "RSTIPOPrompt": RSTIPOPrompt,
    "RSTIPONativeCompletion": RSTIPONativeCompletion,
    "RSTIPONativeExpand": RSTIPONativeExpand,
    "RSTIPONativeControlledExpand": RSTIPONativeControlledExpand,
    "RSTIPONativePipeExpand": RSTIPONativePipeExpand,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RSTIPOPrompt": "RS TIPO Native (Experimental)",
    "RSTIPONativeCompletion": "RS TIPO Native Completion (Legacy)",
    "RSTIPONativeExpand": "RS TIPO Native Expand (Legacy)",
    "RSTIPONativeControlledExpand": "RS TIPO Native Controlled Expand (Legacy)",
    "RSTIPONativePipeExpand": "RS TIPO Native Pipe Expand (Legacy)",
}
