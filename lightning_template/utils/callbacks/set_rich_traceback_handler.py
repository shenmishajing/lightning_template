from typing import Optional

from lightning.pytorch import Callback
from rich.traceback import LOCALS_MAX_LENGTH, LOCALS_MAX_STRING, install


class SetRichTracebackHandlerCallback(Callback):
    def __init__(
        self,
        width: Optional[int] = 100,
        extra_lines: int = 3,
        theme: Optional[str] = None,
        word_wrap: bool = False,
        show_locals: bool = False,
        locals_max_length: int = LOCALS_MAX_LENGTH,
        locals_max_string: int = LOCALS_MAX_STRING,
        locals_hide_dunder: bool = True,
        locals_hide_sunder: Optional[bool] = None,
        indent_guides: bool = True,
        max_frames: int = 100,
    ):
        install(
            width=width,
            extra_lines=extra_lines,
            theme=theme,
            word_wrap=word_wrap,
            show_locals=show_locals,
            locals_max_length=locals_max_length,
            locals_max_string=locals_max_string,
            locals_hide_dunder=locals_hide_dunder,
            locals_hide_sunder=locals_hide_sunder,
            indent_guides=indent_guides,
            max_frames=max_frames,
        )

        self.width = width
        self.extra_lines = extra_lines
        self.theme = theme
        self.word_wrap = word_wrap
        self.show_locals = show_locals
        self.locals_max_length = locals_max_length
        self.locals_max_string = locals_max_string
        self.locals_hide_dunder = locals_hide_dunder
        self.locals_hide_sunder = locals_hide_sunder
        self.indent_guides = indent_guides
        self.max_frames = max_frames
