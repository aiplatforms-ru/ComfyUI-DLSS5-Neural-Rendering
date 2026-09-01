from .nodes import DlssNeuralRenderingExtension


async def comfy_entrypoint() -> DlssNeuralRenderingExtension:
    return DlssNeuralRenderingExtension()


# Lets older registry scanners discover the node name without switching the
# extension back to the legacy API.
if False:
    NODE_CLASS_MAPPINGS = {"Dlss5NeuralRendering": object}

