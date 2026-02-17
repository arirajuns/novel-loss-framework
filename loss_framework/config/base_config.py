"""
Base Configuration Module
Implements abstract base configuration using Template Method pattern
Follows SOLID principles - Single Responsibility and Open/Closed
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import json
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class BaseConfig(ABC):
    """
    Abstract base configuration class using Template Method pattern.
    All specific configurations must inherit from this class.
    """

    def __post_init__(self):
        """Template method for post-initialization validation."""
        self.validate()
        self._freeze = False

    @abstractmethod
    def validate(self) -> None:
        """
        Template method for configuration validation.
        Must be implemented by all subclasses.
        """
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export configuration to JSON format."""
        config_dict = self.to_dict()
        json_str = json.dumps(config_dict, indent=2)

        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                f.write(json_str)

        return json_str

    def to_yaml(self, filepath: Optional[str] = None) -> str:
        """Export configuration to YAML format."""
        config_dict = self.to_dict()
        yaml_str = yaml.dump(config_dict, default_flow_style=False)

        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w") as f:
                f.write(yaml_str)

        return yaml_str

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)

    @classmethod
    def from_json(cls, filepath: str) -> "BaseConfig":
        """Load configuration from JSON file."""
        with open(filepath, "r") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, filepath: str) -> "BaseConfig":
        """Load configuration from YAML file."""
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)

    def freeze(self) -> None:
        """Freeze configuration to prevent modifications."""
        self._freeze = True

    def unfreeze(self) -> None:
        """Unfreeze configuration to allow modifications."""
        self._freeze = False

    def __setattr__(self, name: str, value: Any) -> None:
        """Override setattr to respect frozen state."""
        if getattr(self, "_freeze", False) and name != "_freeze":
            raise AttributeError(
                f"Cannot modify frozen configuration. Field '{name}' is read-only."
            )
        super().__setattr__(name, value)

    def __repr__(self) -> str:
        """String representation of configuration."""
        config_dict = self.to_dict()
        items = [f"{k}={v!r}" for k, v in config_dict.items()]
        return f"{self.__class__.__name__}({', '.join(items)})"

    def copy(self) -> "BaseConfig":
        """Create a deep copy of the configuration."""
        return self.__class__.from_dict(self.to_dict())

    def merge(self, other: "BaseConfig") -> "BaseConfig":
        """Merge another configuration into this one."""
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot merge {type(other)} with {self.__class__}")

        merged_dict = {**self.to_dict(), **other.to_dict()}
        return self.__class__.from_dict(merged_dict)
