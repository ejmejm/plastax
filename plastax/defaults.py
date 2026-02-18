"""Default implementations of user-defined functions for standard backprop + SGD."""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from plastax.states import (
    NeuronState,
    StructureUpdateState,
    tree_replace,
)


# ---------------------------------------------------------------------------
# Default NeuronState class
# ---------------------------------------------------------------------------

class DefaultNeuronState(NeuronState):
    """NeuronState with extra fields for standard backprop."""
    pre_activation: Float[Array, '']
    incoming_activations: Float[Array, 'max_connections']

    def __init__(self, max_connections: int):
        super().__init__(max_connections)
        self.pre_activation = jnp.array(0.0)
        self.incoming_activations = jnp.zeros(max_connections)


# ---------------------------------------------------------------------------
# Default forward function
# ---------------------------------------------------------------------------

def make_default_forward_fn(activation_fn: Callable = jax.nn.relu) -> Callable:
    """Create a forward function that does weighted sum + activation.

    Stores pre_activation and incoming_activations in DefaultNeuronState.
    """
    def forward_fn(
        neuron_state: NeuronState,
        incoming_activations: Float[Array, 'max_connections'],
    ) -> Tuple[Float[Array, ''], NeuronState]:
        weights = neuron_state.weights
        mask = neuron_state.active_connection_mask
        pre_activation = (incoming_activations * weights * mask).sum()
        activation_value = activation_fn(pre_activation)

        updated_state = tree_replace(
            neuron_state,
            activation_value=activation_value,
            pre_activation=pre_activation,
            incoming_activations=incoming_activations * mask,
        )
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
    ) -> Float[Array, '']:
        next_incoming = next_layer_states.incoming_ids
        next_weights = next_layer_states.weights
        next_conn_mask = next_layer_states.active_connection_mask
        next_errors = next_layer_states.error_signal
        next_active = next_layer_states.active_mask

        # Find where this neuron's index appears in next layer's incoming connections
        is_match = (next_incoming == neuron_index) & next_conn_mask

        # For each next-layer neuron, sum the weights of connections from this neuron
        effective_weights = (next_weights * is_match).sum(axis=-1)

        # Weight by the next-layer neurons' error signals and active mask
        error_from_above = (effective_weights * next_errors * next_active).sum()

        return error_from_above

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
        error_signal = neuron_state.error_signal
        pre_activation = neuron_state.pre_activation
        incoming_activations = neuron_state.incoming_activations

        # Activation derivative at pre_activation
        act_deriv = activation_deriv(pre_activation)
        delta = error_signal * act_deriv

        # Weight update: w -= lr * delta * incoming_activation
        conn_mask = neuron_state.active_connection_mask
        weight_grads = delta * incoming_activations * conn_mask
        new_weights = neuron_state.weights - learning_rate * weight_grads

        return tree_replace(
            neuron_state,
            weights=new_weights,
            error_signal=delta,
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
# Connectors and init neuron functions
# ---------------------------------------------------------------------------

def _select_random(
    mask: Bool[Array, 'n'],
    max_connections: int,
    key: jax.Array,
) -> Tuple[Int[Array, 'max_connections'], Bool[Array, 'max_connections']]:
    """Randomly select up to max_connections indices where mask is True.

    Uses shuffle + argsort to be JIT-compatible. Returns (incoming_ids, active_mask).
    """
    n_total = mask.shape[0]
    shuffled = jax.random.permutation(key, n_total)
    sort_keys = jnp.where(mask[shuffled], 0, 1)
    selected = shuffled[jnp.argsort(sort_keys)[:max_connections]]
    is_connected = mask[selected]
    return jnp.where(is_connected, selected, 0), is_connected


def random_connector(
    connectable_mask: Bool[Array, 'total_neurons'],
    index: Int[Array, ''],
    max_connections: int,
    key: jax.Array,
) -> Tuple[Int[Array, 'max_connections'], Bool[Array, 'max_connections']]:
    """Randomly connect to neurons from the entire connectable set."""
    return _select_random(connectable_mask, max_connections, key)


def make_prior_layer_connector(
    n_inputs: int,
    max_hidden_per_layer: int,
) -> Callable:
    """Create a connector that randomly connects to neurons in the prior layer only.

    For layer 0, the prior layer is the inputs. For layer k > 0, it is the
    hidden layer at k-1.
    """
    def connector(
        connectable_mask: Bool[Array, 'total_neurons'],
        index: Int[Array, ''],
        max_connections: int,
        key: jax.Array,
    ) -> Tuple[Int[Array, 'max_connections'], Bool[Array, 'max_connections']]:
        layer_k = (index - n_inputs) // max_hidden_per_layer
        prior_start = jnp.where(layer_k == 0, 0,
                                n_inputs + (layer_k - 1) * max_hidden_per_layer)
        prior_end = jnp.where(layer_k == 0, n_inputs,
                              n_inputs + layer_k * max_hidden_per_layer)

        all_indices = jnp.arange(connectable_mask.shape[0])
        prior_mask = connectable_mask & (all_indices >= prior_start) & (all_indices < prior_end)
        return _select_random(prior_mask, max_connections, key)

    return connector


def make_init_neuron_fn(
    neuron_cls: type[NeuronState],
    connector: Callable,
    weight_init: Callable | None = None,
) -> Callable:
    """Create an init function with pluggable connectivity and weight initialization.

    Args:
        neuron_cls: NeuronState subclass with no-arg constructor.
        connector: Function (connectable_mask, index, max_connections, key)
            -> (incoming_ids, active_connection_mask).
        weight_init: Optional initializer (key, shape, dtype) -> Array.
            Called with shape (max_connections,). Use jax.nn.initializers.normal,
            uniform, etc. Fan-in-based initializers like he_normal require 2D
            shapes and won't work directly. If None, weights default to zeros.

    Returns:
        init_neuron_fn(hidden_states, connectable_mask, index, key) -> NeuronState
    """
    def init_neuron_fn(
        hidden_states: NeuronState,
        connectable_mask: Bool[Array, 'total_neurons'],
        index: Int[Array, ''],
        key: jax.Array,
    ) -> NeuronState:
        state = neuron_cls()
        max_conn = state.weights.shape[0]

        conn_key, weight_key = jax.random.split(key)
        incoming_ids, active_mask = connector(connectable_mask, index, max_conn, conn_key)

        if weight_init is not None:
            weights = weight_init(weight_key, (max_conn,), jnp.float32)
            weights = jnp.where(active_mask, weights, 0.0)
        else:
            weights = jnp.zeros(max_conn)

        return tree_replace(
            state,
            active_mask=jnp.array(True),
            incoming_ids=incoming_ids,
            weights=weights,
            active_connection_mask=active_mask,
        )

    return init_neuron_fn


def make_default_init_neuron_fn(neuron_cls: type[NeuronState]) -> Callable:
    """Create an init function that randomly connects with zero weights.

    Convenience wrapper around make_init_neuron_fn with random_connector
    and no weight initialization.
    """
    return make_init_neuron_fn(neuron_cls, random_connector)


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
