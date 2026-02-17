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

    def __init__(self, max_connections: int):
        self.incoming_ids = jnp.zeros(max_connections, dtype=jnp.int32)
        self.weights = jnp.zeros(max_connections)
        self.active_connection_mask = jnp.zeros(max_connections, dtype=bool)


class ForwardPassState(eqx.Module):
    """State holding forward pass information for a neuron.

    Only requires activation_value. Users subclass to store additional
    information needed by the backward pass (e.g., pre_activation, incoming_activations).
    """
    activation_value: Float[Array, '']

    def __init__(self):
        self.activation_value = jnp.array(0.0)


class BackwardPassState(eqx.Module):
    """State holding backward pass information for a neuron.

    Users can subclass to add extra fields for custom backward passes.
    """
    error_signal: Float[Array, '']

    def __init__(self):
        self.error_signal = jnp.array(0.0)


class NeuronState(eqx.Module):
    """Core state for each neuron. Users subclass to add extra metadata."""
    active_mask: Bool[Array, '']
    connectivity: ConnectivityState
    forward_state: ForwardPassState
    backward_state: BackwardPassState

    def __init__(
        self,
        max_connections: int,
        forward_state: ForwardPassState | None = None,
        backward_state: BackwardPassState | None = None,
    ):
        self.active_mask = jnp.array(False)
        self.connectivity = ConnectivityState(max_connections)
        self.forward_state = forward_state if forward_state is not None else ForwardPassState()
        self.backward_state = backward_state if backward_state is not None else BackwardPassState()


class StructureUpdateState(eqx.Module):
    """State for structure update decisions. Users define all fields."""
    pass


def tree_replace(tree: eqx.Module, **kwargs) -> eqx.Module:
    """Replace fields of an Equinox module by name."""
    values = [kwargs[k] for k in kwargs]
    return eqx.tree_at(lambda x: [getattr(x, k) for k in kwargs], tree, values)
