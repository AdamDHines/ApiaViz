"""Withdrawn full-circle control; historical implementations remain in source.zip.

Stale callers fail explicitly rather than silently running another controller.
"""


class PanoramicConfirmationController:
    def __init__(self, calibration, settings=None, phase=1):
        raise RuntimeError('Full-circle scanning was removed at the user\'s request. Use bounded controller settings; historical replay requires the frozen source archive.')
