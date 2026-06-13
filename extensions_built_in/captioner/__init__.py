from toolkit.extension import Extension


class Qwen3VLCaptionerExtension(Extension):
    uid = "Qwen3VLCaptioner"
    name = "Qwen 3VL Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .Qwen3VLCaptioner import Qwen3VLCaptioner

        return Qwen3VLCaptioner


class Ideogram4CaptionerExtension(Extension):
    uid = "Ideogram4Captioner"
    name = "Ideogram4 Captioner"

    @classmethod
    def get_process(cls):
        # import your process class here so it is only loaded when needed and return it
        from .Ideogram4Captioner import Ideogram4Captioner

        return Ideogram4Captioner


AI_TOOLKIT_EXTENSIONS = [
    Qwen3VLCaptionerExtension,
    Ideogram4CaptionerExtension,
]
