from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


class ConnectivityState(eqx.Module):
    """State holding a neuron's incoming connection information.

    Users can subclass to add extra per-connection metadata.
    """
    incoming_ids: Int[Array, 'max_connections']
    weights: Float[Array, 'max_connections']
    active_connection_mask: Bool[Array, 'max_connections']


class ForwardPassState(eqx.Module):
    """State holding forward pass information for a neuron.

    Only requires activation_value. Users subclass to store additional
    information needed by the backward pass (e.g., pre_activation, incoming_activations).
    """
    activation_value: Float[Array, '']


class BackwardPassState(eqx.Module):
    """State holding backward pass information for a neuron.

    Users can subclass to add extra fields for custom backward passes.
    """
    error_signal: Float[Array, '']


class NeuronState(eqx.Module):
    """Core state for each neuron. Users subclass to add extra metadata."""
    active_mask: Bool[Array, '']
    connectivity: ConnectivityState
    forward_state: ForwardPassState
    backward_state: BackwardPassState


class StructureUpdateState(eqx.Module):
    """State for structure update decisions. Users define all fields."""
    pass


def tree_replace(tree: eqx.Module, **kwargs) -> eqx.Module:
    """Replace fields of an Equinox module by name."""
    values = [kwargs[k] for k in kwargs]
    return eqx.tree_at(lambda x: [getattr(x, k) for k in kwargs], tree, values)
