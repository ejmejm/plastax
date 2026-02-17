"""Default implementations of user-defined functions for standard backprop + SGD."""

from functools import partial
from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from plastax.states import (
    BackwardPassState,
    ConnectivityState,
    ForwardPassState,
    NeuronState,
    StructureUpdateState,
    tree_replace,
)


# ---------------------------------------------------------------------------
# Default forward pass state (extends base with fields needed for backprop)
# ---------------------------------------------------------------------------

class DefaultForwardPassState(ForwardPassState):
    pre_activation: Float[Array, '']
    incoming_activations: Float[Array, 'max_connections']

    def __init__(self, max_connections: int):
        super().__init__()
        self.pre_activation = jnp.array(0.0)
        self.incoming_activations = jnp.zeros(max_connections)


# ---------------------------------------------------------------------------
# Default NeuronState class
# ---------------------------------------------------------------------------

class DefaultNeuronState(NeuronState):
    """NeuronState using DefaultForwardPassState for standard backprop."""

    def __init__(self, max_connections: int):
        super().__init__(max_connections, forward_state=DefaultForwardPassState(max_connections))


# ---------------------------------------------------------------------------
# Default forward function
# ---------------------------------------------------------------------------

def make_default_forward_fn(activation_fn: Callable = jax.nn.relu) -> Callable:
    """Create a forward function that does weighted sum + activation.

    Stores pre_activation and incoming_activations in DefaultForwardPassState.
    """
    def forward_fn(
        neuron_state: NeuronState,
        incoming_activations: Float[Array, 'max_connections'],
    ) -> Tuple[Float[Array, ''], NeuronState]:
        weights = neuron_state.connectivity.weights
        mask = neuron_state.connectivity.active_connection_mask
        pre_activation = (incoming_activations * weights * mask).sum()
        activation_value = activation_fn(pre_activation)

        updated_forward = tree_replace(
            neuron_state.forward_state,
            activation_value=activation_value,
            pre_activation=pre_activation,
            incoming_activations=incoming_activations * mask,
        )
        updated_state = tree_replace(neuron_state, forward_state=updated_forward)
        return activation_value, updated_state

    return forward_fn


# ---------------------------------------------------------------------------
# Default backward signal function
# ---------------------------------------------------------------------------

def make_default_backward_signal_fn() -> Callable:
    """Create a backward signal function that propagates error via outgoing connections.

    For each neuron, finds which next-layer neurons connect from it and sums
    (weight * error_signal * connection_mask) to get the incoming error.
    """
    def backward_signal_fn(
        neuron_state: NeuronState,
        neuron_index: Int[Array, ''],
        next_layer_states: NeuronState,
    ) -> BackwardPassState:
        # next_layer_states.connectivity.incoming_ids: (next_layer_size, max_conn)
        next_incoming = next_layer_states.connectivity.incoming_ids
        next_weights = next_layer_states.connectivity.weights
        next_conn_mask = next_layer_states.connectivity.active_connection_mask
        next_errors = next_layer_states.backward_state.error_signal  # (next_layer_size,)
        next_active = next_layer_states.active_mask  # (next_layer_size,)

        # Find where this neuron's index appears in next layer's incoming connections
        is_match = (next_incoming == neuron_index) & next_conn_mask  # (next_layer_size, max_conn)

        # For each next-layer neuron, sum the weights of connections from this neuron
        effective_weights = (next_weights * is_match).sum(axis=-1)  # (next_layer_size,)

        # Weight by the next-layer neurons' error signals and active mask
        error_from_above = (effective_weights * next_errors * next_active).sum()

        return tree_replace(neuron_state.backward_state, error_signal=error_from_above)

    return backward_signal_fn


# ---------------------------------------------------------------------------
# Default neuron update function
# ---------------------------------------------------------------------------

def make_default_neuron_update_fn(
    learning_rate: float,
    activation_fn: Callable = jax.nn.relu,
) -> Callable:
    """Create a neuron update function that applies activation derivative and SGD.

    Reads error_signal (set by backward_signal_fn), applies activation derivative
    to get delta, updates weights via SGD, and overwrites error_signal with delta.
    """
    # Compute activation derivative: jax.grad on a scalar-to-scalar function
    activation_deriv = jax.grad(activation_fn)

    def neuron_update_fn(neuron_state: NeuronState) -> NeuronState:
        error_signal = neuron_state.backward_state.error_signal
        pre_activation = neuron_state.forward_state.pre_activation
        incoming_activations = neuron_state.forward_state.incoming_activations

        # Activation derivative at pre_activation
        act_deriv = activation_deriv(pre_activation)
        delta = error_signal * act_deriv

        # Weight update: w -= lr * delta * incoming_activation
        conn_mask = neuron_state.connectivity.active_connection_mask
        weight_grads = delta * incoming_activations * conn_mask
        new_weights = neuron_state.connectivity.weights - learning_rate * weight_grads

        updated_connectivity = tree_replace(neuron_state.connectivity, weights=new_weights)
        updated_backward = tree_replace(neuron_state.backward_state, error_signal=delta)

        return tree_replace(
            neuron_state,
            connectivity=updated_connectivity,
            backward_state=updated_backward,
        )

    return neuron_update_fn


# ---------------------------------------------------------------------------
# Default structure update (no-op)
# ---------------------------------------------------------------------------

def default_structure_update_fn(
    layer_states: NeuronState,
    next_layer_states: NeuronState,
    structure_state: StructureUpdateState,
) -> Tuple[StructureUpdateState, Bool[Array, 'layer_size'], Int[Array, '']]:
    layer_size = layer_states.active_mask.shape[0]
    prune_mask = jnp.zeros(layer_size, dtype=bool)
    n_generate = jnp.array(0, dtype=jnp.int32)
    return structure_state, prune_mask, n_generate


# ---------------------------------------------------------------------------
# Default init neuron function
# ---------------------------------------------------------------------------

def make_default_init_neuron_fn(neuron_cls: type[NeuronState]) -> Callable:
    """Create an init function that returns a zeroed NeuronState with the given connectivity."""
    def init_neuron_fn(
        connectivity: ConnectivityState,
        index: Int[Array, ''],
        key: jax.Array,
    ) -> NeuronState:
        state = neuron_cls()
        return tree_replace(state, active_mask=jnp.array(True), connectivity=connectivity)
    return init_neuron_fn


# ---------------------------------------------------------------------------
# Default output error function (MSE derivative)
# ---------------------------------------------------------------------------

def make_default_output_error_fn() -> Callable:
    """MSE derivative: 2 * (prediction - target) / n_outputs."""
    def compute_output_error(
        output_activations: Float[Array, 'n_outputs'],
        targets: Float[Array, 'n_outputs'],
    ) -> Float[Array, 'n_outputs']:
        n = output_activations.shape[0]
        return 2.0 * (output_activations - targets) / n

    return compute_output_error
