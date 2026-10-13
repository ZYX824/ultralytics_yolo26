"""Errors shared by the inference pipeline and the HTTP adapters."""


class GPUOutOfMemory(RuntimeError):
    """The inference process stopped after GPU memory exhaustion."""


class ServiceBusy(RuntimeError):
    """The bounded inference or batch queue is full."""
