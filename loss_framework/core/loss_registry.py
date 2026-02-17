"""
Loss Registry Module
Implements Registry pattern for loss function management
Provides centralized registration and lookup of loss functions
"""

from typing import Dict, Type, Optional, Callable
import inspect


class LossRegistry:
    """
    Registry pattern for managing loss function classes.

    Provides centralized registration and lookup, enabling:
    - Dynamic loss function creation by name
    - Plugin-style architecture
    - Easy extension with new loss functions

    Singleton pattern implementation to ensure single registry instance.
    """

    _instance: Optional["LossRegistry"] = None
    _registry: Dict[str, Type] = {}
    _metadata: Dict[str, Dict] = {}

    def __new__(cls) -> "LossRegistry":
        """Singleton pattern - ensure only one registry exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._registry = {}
            cls._metadata = {}
        return cls._instance

    @classmethod
    def register(cls, name: Optional[str] = None, **metadata) -> Callable:
        """
        Decorator to register a loss function class.

        Args:
            name: Optional custom name for the loss
            **metadata: Additional metadata about the loss

        Example:
            @LossRegistry.register(name="custom_loss", category="classification")
            class CustomLoss(BaseLoss):
                pass
        """

        def decorator(loss_class: Type) -> Type:
            # Use class name if no custom name provided
            loss_name = name if name is not None else loss_class.__name__

            # Register the class
            cls._registry[loss_name] = loss_class

            # Store metadata
            cls._metadata[loss_name] = {
                "class": loss_class,
                "name": loss_name,
                "docstring": loss_class.__doc__,
                "module": loss_class.__module__,
                **metadata,
            }

            return loss_class

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Type]:
        """
        Get a loss function class by name.

        Args:
            name: Name of the registered loss function

        Returns:
            The loss function class, or None if not found
        """
        return cls._registry.get(name)

    @classmethod
    def create(cls, name: str, *args, **kwargs):
        """
        Create an instance of a registered loss function.

        Args:
            name: Name of the registered loss function
            *args: Positional arguments for loss constructor
            **kwargs: Keyword arguments for loss constructor

        Returns:
            Instance of the loss function

        Raises:
            ValueError: If loss function not found
        """
        loss_class = cls.get(name)
        if loss_class is None:
            raise ValueError(
                f"Loss function '{name}' not found. "
                f"Available: {list(cls._registry.keys())}"
            )

        return loss_class(*args, **kwargs)

    @classmethod
    def list_losses(cls, category: Optional[str] = None) -> list:
        """
        List all registered loss functions.

        Args:
            category: Optional filter by category

        Returns:
            List of registered loss function names
        """
        if category is None:
            return list(cls._registry.keys())

        return [
            name
            for name, meta in cls._metadata.items()
            if meta.get("category") == category
        ]

    @classmethod
    def get_metadata(cls, name: str) -> Optional[Dict]:
        """Get metadata for a registered loss function."""
        return cls._metadata.get(name)

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a loss function.

        Args:
            name: Name of loss function to unregister

        Returns:
            True if unregistered, False if not found
        """
        if name in cls._registry:
            del cls._registry[name]
            del cls._metadata[name]
            return True
        return False

    @classmethod
    def clear(cls) -> None:
        """Clear all registered loss functions."""
        cls._registry.clear()
        cls._metadata.clear()

    @classmethod
    def get_categories(cls) -> list:
        """Get all unique categories of registered losses."""
        categories = set()
        for meta in cls._metadata.values():
            if "category" in meta:
                categories.add(meta["category"])
        return sorted(list(categories))

    @classmethod
    def info(cls) -> Dict:
        """Get registry information."""
        return {
            "registered_losses": list(cls._registry.keys()),
            "count": len(cls._registry),
            "categories": cls.get_categories(),
        }


# Convenience function for registration
def register_loss(name: Optional[str] = None, **metadata):
    """
    Convenience decorator for registering loss functions.

    Example:
        @register_loss(name="my_loss", category="classification")
        class MyLoss(BaseLoss):
            pass
    """
    return LossRegistry.register(name=name, **metadata)


# Function registry for functional loss implementations
class FunctionalLossRegistry:
    """Registry for functional (stateless) loss implementations."""

    _instance: Optional["FunctionalLossRegistry"] = None
    _registry: Dict[str, Callable] = {}

    def __new__(cls) -> "FunctionalLossRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._registry = {}
        return cls._instance

    @classmethod
    def register(cls, name: str) -> Callable:
        """Register a functional loss implementation."""

        def decorator(func: Callable) -> Callable:
            cls._registry[name] = func
            return func

        return decorator

    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """Get functional loss by name."""
        return cls._registry.get(name)

    @classmethod
    def list_functions(cls) -> list:
        """List all registered functional losses."""
        return list(cls._registry.keys())


# Convenience function
def register_functional_loss(name: str):
    """Decorator for registering functional losses."""
    return FunctionalLossRegistry.register(name)
