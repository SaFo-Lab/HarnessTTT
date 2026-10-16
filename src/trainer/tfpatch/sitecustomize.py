"""Compatibility shims for the container's Transformers/Accelerate versions."""

try:
    from transformers import masking_utils as _masking_utils

    _original_find = _masking_utils.find_packed_sequence_indices

    def _guarded_find_packed_sequence_indices(position_ids):
        if position_ids is None or position_ids.shape[-1] == 0:
            return None
        return _original_find(position_ids)

    _masking_utils.find_packed_sequence_indices = _guarded_find_packed_sequence_indices
except Exception:
    pass

try:
    from contextlib import contextmanager as _contextmanager

    import accelerate as _accelerate
    from accelerate import big_modeling as _big_modeling
    import torch.nn as _nn

    _original_init_on_device = _big_modeling.init_on_device

    @_contextmanager
    def _compatible_init_on_device(*args, **kwargs):
        original_new = _nn.Parameter.__new__

        def _parameter_new(cls, data=None, requires_grad=True, **attributes):
            parameter = original_new(cls, data, requires_grad)
            parameter.__dict__.update(attributes)
            return parameter

        _nn.Parameter.__new__ = staticmethod(_parameter_new)
        try:
            with _original_init_on_device(*args, **kwargs) as value:
                yield value
        finally:
            _nn.Parameter.__new__ = original_new

    _big_modeling.init_on_device = _compatible_init_on_device
    _accelerate.init_on_device = _compatible_init_on_device
except Exception:
    pass
