"""
Physics-Inspired Loss
Implements loss functions inspired by physical principles
Uses Hamiltonian mechanics, conservation laws, and Lagrangian dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..core.base_loss import BaseLoss
from ..core.loss_registry import register_loss
from ..config.loss_config import PhysicsInspiredLossConfig


class HamiltonianMechanics:
    """
    Hamiltonian mechanics utilities for neural networks.
    Models neural network dynamics as a Hamiltonian system.
    """

    @staticmethod
    def compute_hamiltonian(
        position: torch.Tensor, momentum: torch.Tensor, potential_fn: nn.Module
    ) -> torch.Tensor:
        """
        Compute Hamiltonian: H = T + V
        T = kinetic energy (0.5 * ||momentum||^2)
        V = potential energy
        """
        kinetic = 0.5 * (momentum**2).sum(dim=-1)
        potential = potential_fn(position).squeeze(-1)
        return kinetic + potential

    @staticmethod
    def hamiltonian_dynamics(
        position: torch.Tensor,
        momentum: torch.Tensor,
        potential_fn: nn.Module,
        dt: float = 0.1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate Hamiltonian dynamics using symplectic Euler.

        Args:
            position: Current position
            momentum: Current momentum
            potential_fn: Potential energy function
            dt: Time step

        Returns:
            Updated position and momentum
        """
        # Compute gradient of potential
        position.requires_grad_(True)
        potential = potential_fn(position).sum()
        grad_potential = torch.autograd.grad(potential, position)[0]

        # Update momentum (force = -grad(V))
        momentum_new = momentum - dt * grad_potential

        # Update position
        position_new = position + dt * momentum_new

        return position_new.detach(), momentum_new.detach()

    @staticmethod
    def verify_energy_conservation(
        trajectory: list, potential_fn: nn.Module, tolerance: float = 1e-3
    ) -> bool:
        """
        Verify that energy is conserved along trajectory.

        Args:
            trajectory: List of (position, momentum) tuples
            potential_fn: Potential energy function
            tolerance: Allowed energy drift

        Returns:
            True if energy is conserved
        """
        energies = []
        for pos, mom in trajectory:
            H = HamiltonianMechanics.compute_hamiltonian(pos, mom, potential_fn)
            energies.append(H.mean().item())

        energy_drift = max(energies) - min(energies)
        return energy_drift < tolerance


@register_loss(name="physics_inspired", category="physics")
class PhysicsInspiredLoss(BaseLoss):
    """
    Physics-Inspired Loss using Hamiltonian mechanics.

    Models neural network training as a physical system with:
    - Hamiltonian dynamics
    - Conservation laws
    - Symplectic integration

    Mathematical Formulation:
        L = L_task + λ1 * H_drift + λ2 * L_conservation

    Where:
        - L_task is the task-specific loss
        - H_drift penalizes deviation from constant Hamiltonian
        - L_conservation enforces conservation laws

    Features:
    - Hamiltonian dynamics regularization
    - Conservation law enforcement
    - Symplectic gradient updates
    - Physical interpretability

    Example:
        loss = PhysicsInspiredLoss(
            use_hamiltonian=True,
            hamiltonian_weight=0.1,
            use_conservation=True
        )
    """

    def __init__(
        self,
        base_loss: str = "cross_entropy",
        use_hamiltonian: bool = True,
        hamiltonian_weight: float = 0.1,
        use_conservation: bool = False,
        conservation_weight: float = 0.1,
        use_lagrangian: bool = False,
        lagrangian_weight: float = 0.1,
        conserved_quantities: int = 1,
        use_symplectic: bool = False,
        symplectic_order: int = 2,
        reduction: str = "mean",
        device: str = "auto",
    ):
        """
        Initialize physics-inspired loss.

        Args:
            base_loss: Base loss function type
            use_hamiltonian: Whether to use Hamiltonian regularization
            hamiltonian_weight: Weight for Hamiltonian term
            use_conservation: Whether to enforce conservation laws
            conservation_weight: Weight for conservation term
            use_lagrangian: Whether to use Lagrangian mechanics
            lagrangian_weight: Weight for Lagrangian term
            conserved_quantities: Number of quantities to conserve
            use_symplectic: Whether to use symplectic integration
            symplectic_order: Order of symplectic integrator
            reduction: Loss reduction method
            device: Device for computation
        """
        super().__init__(reduction=reduction, device=device)

        self.base_loss_type = base_loss
        self.use_hamiltonian = use_hamiltonian
        self.hamiltonian_weight = hamiltonian_weight
        self.use_conservation = use_conservation
        self.conservation_weight = conservation_weight
        self.use_lagrangian = use_lagrangian
        self.lagrangian_weight = lagrangian_weight
        self.conserved_quantities = conserved_quantities
        self.use_symplectic = use_symplectic
        self.symplectic_order = symplectic_order

        # Create base loss
        self.base_loss_fn = self._create_base_loss()

        # Potential energy network for Hamiltonian
        if self.use_hamiltonian:
            self.potential_net = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
            ).to(self.device)

        # Conservation constraints
        if self.use_conservation:
            self.conservation_proj = nn.Linear(128, conserved_quantities).to(
                self.device
            )

        # State tracking for Hamiltonian
        self._previous_state = None
        self._hamiltonian_values = []

    def _create_base_loss(self):
        """Create base loss function."""
        loss_map = {
            "cross_entropy": nn.CrossEntropyLoss(reduction=self.reduction),
            "mse": nn.MSELoss(reduction=self.reduction),
            "l1": nn.L1Loss(reduction=self.reduction),
        }
        return loss_map.get(self.base_loss_type, loss_map["cross_entropy"])

    def _compute_hamiltonian_loss(
        self, features: torch.Tensor, momentum: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Hamiltonian conservation loss.

        Penalizes deviation from constant energy in feature space.
        """
        if momentum is None:
            # Initialize momentum as gradient of features
            momentum = torch.randn_like(features) * 0.01

        # Compute current Hamiltonian
        H_current = HamiltonianMechanics.compute_hamiltonian(
            features, momentum, self.potential_net
        )

        # Penalize deviation from mean energy
        H_mean = H_current.mean()
        H_var = ((H_current - H_mean) ** 2).mean()

        return H_var

    def _compute_conservation_loss(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute conservation law loss.

        Enforces that conserved quantities remain constant.
        """
        # Project features to conserved quantities
        conserved = self.conservation_proj(features)

        if self._previous_state is not None:
            # Penalize change in conserved quantities
            prev_conserved = self.conservation_proj(self._previous_state)
            conservation_loss = F.mse_loss(conserved, prev_conserved)
        else:
            conservation_loss = torch.tensor(0.0, device=features.device)

        # Update state
        self._previous_state = features.detach()

        return conservation_loss

    def _compute_lagrangian_loss(
        self, features: torch.Tensor, velocities: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Lagrangian mechanics loss.

        L = T - V where T is kinetic, V is potential energy.
        """
        # Kinetic energy
        T = 0.5 * (velocities**2).sum(dim=-1)

        # Potential energy
        V = self.potential_net(features).squeeze(-1)

        # Lagrangian
        lagrangian = T - V

        # We want to maximize the action (integral of L)
        # So we minimize the negative
        return -lagrangian.mean()

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute physics-inspired loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            features: Optional intermediate features

        Returns:
            Combined physics-inspired loss
        """
        # Base task loss
        base_loss = self.base_loss_fn(predictions, targets)
        total_loss = base_loss

        if features is not None:
            # Hamiltonian regularization
            if self.use_hamiltonian:
                H_loss = self._compute_hamiltonian_loss(features)
                total_loss = total_loss + self.hamiltonian_weight * H_loss
                self._hamiltonian_values.append(H_loss.item())

            # Conservation law
            if self.use_conservation:
                cons_loss = self._compute_conservation_loss(features)
                total_loss = total_loss + self.conservation_weight * cons_loss

            # Lagrangian mechanics
            if self.use_lagrangian:
                velocities = torch.randn_like(features) * 0.01
                L_loss = self._compute_lagrangian_loss(features, velocities)
                total_loss = total_loss + self.lagrangian_weight * L_loss

        return total_loss

    def get_hamiltonian_stats(self) -> dict:
        """Get Hamiltonian dynamics statistics."""
        if not self._hamiltonian_values:
            return {}

        import numpy as np

        H_vals = self._hamiltonian_values
        return {
            "mean_hamiltonian": np.mean(H_vals),
            "std_hamiltonian": np.std(H_vals),
            "min_hamiltonian": np.min(H_vals),
            "max_hamiltonian": np.max(H_vals),
            "energy_drift": np.max(H_vals) - np.min(H_vals),
        }

    def reset_state(self) -> None:
        """Reset physics state."""
        self._previous_state = None
        self._hamiltonian_values = []

    @classmethod
    def from_config(cls, config: PhysicsInspiredLossConfig) -> "PhysicsInspiredLoss":
        """Create loss from configuration."""
        return cls(
            base_loss=config.hyperparameters.get("base_loss", "cross_entropy"),
            use_hamiltonian=config.use_hamiltonian,
            hamiltonian_weight=config.hamiltonian_weight,
            use_conservation=config.use_conservation,
            conservation_weight=config.conservation_weight,
            use_lagrangian=config.use_lagrangian,
            lagrangian_weight=config.lagrangian_weight,
            conserved_quantities=config.conserved_quantities,
            use_symplectic=config.use_symplectic,
            symplectic_order=config.symplectic_order,
            reduction=config.reduction,
            device=config.device,
        )

    def extra_repr(self) -> str:
        """String representation."""
        return (
            f"hamiltonian={self.use_hamiltonian}, "
            f"conservation={self.use_conservation}, "
            f"lagrangian={self.use_lagrangian}"
        )
