"""
Input Validation Module
Provides validation utilities for loss function inputs
"""

import torch
from typing import Tuple, Optional, List, Union


class InputValidator:
    """
    Validation utilities for loss function inputs.
    Provides comprehensive validation with informative error messages.
    """

    @staticmethod
    def validate_shape(
        tensor: torch.Tensor,
        expected_shape: Optional[Tuple[int, ...]] = None,
        min_dims: Optional[int] = None,
        max_dims: Optional[int] = None,
        name: str = "tensor",
    ) -> None:
        """
        Validate tensor shape.

        Args:
            tensor: Tensor to validate
            expected_shape: Expected shape (None for any)
            min_dims: Minimum number of dimensions
            max_dims: Maximum number of dimensions
            name: Name of tensor for error messages
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")

        if expected_shape is not None:
            if tensor.shape != expected_shape:
                raise ValueError(
                    f"{name} has shape {tensor.shape}, expected {expected_shape}"
                )

        if min_dims is not None and len(tensor.shape) < min_dims:
            raise ValueError(
                f"{name} has {len(tensor.shape)} dimensions, "
                f"minimum required: {min_dims}"
            )

        if max_dims is not None and len(tensor.shape) > max_dims:
            raise ValueError(
                f"{name} has {len(tensor.shape)} dimensions, "
                f"maximum allowed: {max_dims}"
            )

    @staticmethod
    def validate_type(
        tensor: torch.Tensor,
        expected_dtype: Union[torch.dtype, List[torch.dtype]],
        name: str = "tensor",
    ) -> None:
        """
        Validate tensor data type.

        Args:
            tensor: Tensor to validate
            expected_dtype: Expected dtype or list of dtypes
            name: Name of tensor for error messages
        """
        if isinstance(expected_dtype, list):
            if tensor.dtype not in expected_dtype:
                raise TypeError(
                    f"{name} has dtype {tensor.dtype}, expected one of {expected_dtype}"
                )
        else:
            if tensor.dtype != expected_dtype:
                raise TypeError(
                    f"{name} has dtype {tensor.dtype}, expected {expected_dtype}"
                )

    @staticmethod
    def validate_range(
        tensor: torch.Tensor,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        strict_min: bool = False,
        strict_max: bool = False,
        name: str = "tensor",
    ) -> None:
        """
        Validate tensor value range.

        Args:
            tensor: Tensor to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            strict_min: Whether minimum is strict (>) vs inclusive (>=)
            strict_max: Whether maximum is strict (<) vs inclusive (<=)
            name: Name of tensor for error messages
        """
        if min_val is not None:
            if strict_min:
                if tensor.min() <= min_val:
                    raise ValueError(
                        f"{name} has values <= {min_val}, but must be strictly greater"
                    )
            else:
                if tensor.min() < min_val:
                    raise ValueError(
                        f"{name} has values < {min_val}, "
                        f"but minimum allowed is {min_val}"
                    )

        if max_val is not None:
            if strict_max:
                if tensor.max() >= max_val:
                    raise ValueError(
                        f"{name} has values >= {max_val}, but must be strictly less"
                    )
            else:
                if tensor.max() > max_val:
                    raise ValueError(
                        f"{name} has values > {max_val}, "
                        f"but maximum allowed is {max_val}"
                    )

    @staticmethod
    def validate_probabilities(
        tensor: torch.Tensor,
        dim: int = -1,
        tolerance: float = 1e-6,
        name: str = "tensor",
    ) -> None:
        """
        Validate that tensor contains valid probabilities.

        Args:
            tensor: Tensor to validate
            dim: Dimension along which probabilities should sum to 1
            tolerance: Tolerance for sum check
            name: Name of tensor for error messages
        """
        # Check range [0, 1]
        InputValidator.validate_range(
            tensor, min_val=0, max_val=1, strict_min=False, strict_max=False, name=name
        )

        # Check sum to 1
        prob_sum = tensor.sum(dim=dim)
        if not torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=tolerance):
            raise ValueError(
                f"{name} probabilities do not sum to 1 along dimension {dim}. "
                f"Max deviation: {(prob_sum - 1).abs().max().item()}"
            )

    @staticmethod
    def validate_logits(tensor: torch.Tensor, name: str = "tensor") -> None:
        """
        Validate that tensor contains valid logits.

        Args:
            tensor: Tensor to validate
            name: Name of tensor for error messages
        """
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name} contains non-finite values (inf or nan)")

    @staticmethod
    def validate_matching_shapes(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        broadcastable: bool = False,
        name1: str = "tensor1",
        name2: str = "tensor2",
    ) -> None:
        """
        Validate that two tensors have matching shapes.

        Args:
            tensor1: First tensor
            tensor2: Second tensor
            broadcastable: Whether shapes must be exactly equal or broadcastable
            name1: Name of first tensor
            name2: Name of second tensor
        """
        if not broadcastable:
            if tensor1.shape != tensor2.shape:
                raise ValueError(
                    f"Shape mismatch: {name1} has shape {tensor1.shape}, "
                    f"{name2} has shape {tensor2.shape}"
                )
        else:
            try:
                torch.broadcast_shapes(tensor1.shape, tensor2.shape)
            except RuntimeError:
                raise ValueError(
                    f"Shapes {tensor1.shape} and {tensor2.shape} are not broadcastable"
                )

    @staticmethod
    def validate_same_device(
        tensor1: torch.Tensor,
        tensor2: torch.Tensor,
        name1: str = "tensor1",
        name2: str = "tensor2",
    ) -> None:
        """
        Validate that two tensors are on the same device.

        Args:
            tensor1: First tensor
            tensor2: Second tensor
            name1: Name of first tensor
            name2: Name of second tensor
        """
        if tensor1.device != tensor2.device:
            raise ValueError(
                f"Device mismatch: {name1} is on {tensor1.device}, "
                f"{name2} is on {tensor2.device}"
            )

    @staticmethod
    def validate_classification_targets(
        targets: torch.Tensor, num_classes: Optional[int] = None, name: str = "targets"
    ) -> None:
        """
        Validate classification target labels.

        Args:
            targets: Target tensor
            num_classes: Expected number of classes
            name: Name of tensor for error messages
        """
        if targets.dtype not in [torch.int64, torch.long, torch.int32]:
            raise TypeError(
                f"{name} must be integer type for classification, got {targets.dtype}"
            )

        if num_classes is not None:
            min_label = targets.min().item()
            max_label = targets.max().item()

            if min_label < 0:
                raise ValueError(f"{name} contains negative labels: {min_label}")

            if max_label >= num_classes:
                raise ValueError(
                    f"{name} contains label {max_label}, "
                    f"but num_classes is {num_classes}"
                )

    @staticmethod
    def validate_batch_size(
        *tensors: torch.Tensor, name: str = "batch dimension"
    ) -> int:
        """
        Validate that all tensors have the same batch size.

        Args:
            *tensors: Tensors to validate
            name: Name for error messages

        Returns:
            The common batch size
        """
        if len(tensors) == 0:
            return 0

        batch_size = tensors[0].shape[0]

        for i, tensor in enumerate(tensors[1:], 1):
            if tensor.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: tensor 0 has batch size {batch_size}, "
                    f"tensor {i} has batch size {tensor.shape[0]}"
                )

        return batch_size

    @staticmethod
    def sanitize_inputs(
        predictions: torch.Tensor, targets: torch.Tensor, epsilon: float = 1e-7
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sanitize inputs by replacing NaN/Inf values.

        Args:
            predictions: Prediction tensor
            targets: Target tensor
            epsilon: Small value to replace problematic values

        Returns:
            Tuple of sanitized tensors
        """
        predictions = torch.where(
            torch.isfinite(predictions), predictions, torch.zeros_like(predictions)
        )

        targets = torch.where(
            torch.isfinite(targets), targets, torch.zeros_like(targets)
        )

        return predictions, targets
