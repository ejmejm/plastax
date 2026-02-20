"""Standard backprop + SGD implementations of user-defined functions."""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from plastax.network import StateUpdateFunctions
from plastax.states import (
    CONNECTION_PADDING,
    NeuronState,
    StructureUpdateState,
    tree_replace,
)


# ---------------------------------------------------------------------------
# Backprop NeuronState class
# ---------------------------------------------------------------------------

class BackpropNeuronState(NeuronState):
    """NeuronState with extra fields for standard backprop."""
    pre_activation: Float[Array, '']
    incoming_activations: Float[Array, 'max_connections']

    def __init__(self, max_connections: int):
        super().__init__(max_connections)
        self.pre_activation = jnp.array(0.0)
        self.incoming_activations = jnp.zeros(max_connections)


# ---------------------------------------------------------------------------
# Weighted-sum forward function
# ---------------------------------------------------------------------------

def make_weighted_sum_forward_fn(activation_fn: Callable = jax.nn.relu) -> Callable:
    """Create a forward function that does weighted sum + activation.

    Stores pre_activation and incoming_activations in BackpropNeuronState.
    """
    def forward_fn(
        neuron_state: NeuronState,
        incoming_activations: Float[Array, 'max_connections'],
    ) -> Tuple[Float[Array, ''], NeuronState]:
        weights = neuron_state.weights
        mask = neuron_state.get_active_connection_mask()
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
# Backprop error signal function
# ---------------------------------------------------------------------------

def make_backprop_error_signal_fn(
    activation_deriv: Callable = jax.grad(jax.nn.relu),
) -> Callable:
    """Create a backward signal function that propagates error via outgoing connections.

    For each neuron, finds which next-layer neurons connect from it and computes
    delta_j = error_j * activation_deriv(pre_activation_j) for each sending
    neuron j, then sums (weight * delta * connection_mask) to get the error.

    Args:
        activation_deriv: Derivative of the sending layer's activation function.
            Called as activation_deriv(pre_activation) -> scalar.
    """
    def backward_signal_fn(
        neuron_state: NeuronState,
        neuron_index: Int[Array, ''],
        next_layer_states: NeuronState,
    ) -> Float[Array, '']:
        next_incoming = next_layer_states.incoming_ids
        next_weights = next_layer_states.weights
        next_conn_mask = next_layer_states.get_active_connection_mask()
        next_errors = next_layer_states.error_signal
        next_active = next_layer_states.active_mask

        # Compute delta for each sending neuron: error * f'(pre_activation)
        next_deltas = next_errors * jax.vmap(activation_deriv)(next_layer_states.pre_activation)

        # Find where this neuron's index appears in next layer's incoming connections
        is_match = (next_incoming == neuron_index) & next_conn_mask

        # For each next-layer neuron, sum the weights of connections from this neuron
        effective_weights = (next_weights * is_match).sum(axis=-1)

        # Weight by deltas and active mask
        error_from_above = (effective_weights * next_deltas * next_active).sum()

        return error_from_above

    return backward_signal_fn


# ---------------------------------------------------------------------------
# SGD neuron update function
# ---------------------------------------------------------------------------

def make_sgd_update_fn(
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
        conn_mask = neuron_state.get_active_connection_mask()
        weight_grads = delta * incoming_activations * conn_mask
        new_weights = neuron_state.weights - learning_rate * weight_grads

        return tree_replace(
            neuron_state,
            weights=new_weights,
            error_signal=delta,
        )

    return neuron_update_fn


# ---------------------------------------------------------------------------
# No-op structure update
# ---------------------------------------------------------------------------

def noop_structure_update_fn(
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
) -> Int[Array, 'max_connections']:
    """Randomly select up to max_connections indices where mask is True.

    Uses shuffle + argsort to be JIT-compatible. Returns incoming_ids.
    """
    n_total = mask.shape[0]
    shuffled = jax.random.permutation(key, n_total)
    sort_keys = jnp.where(mask[shuffled], 0, 1)
    selected = shuffled[jnp.argsort(sort_keys)[:max_connections]]
    is_connected = mask[selected]
    return jnp.where(is_connected, selected, CONNECTION_PADDING)


def random_connector(
    connectable_mask: Bool[Array, 'total_neurons'],
    index: Int[Array, ''],
    max_connections: int,
    key: jax.Array,
) -> Int[Array, 'max_connections']:
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
    ) -> Int[Array, 'max_connections']:
        layer_k = (index - n_inputs) // max_hidden_per_layer
        prior_start = jnp.where(layer_k == 0, 0,
                                n_inputs + (layer_k - 1) * max_hidden_per_layer)
        prior_end = jnp.where(layer_k == 0, n_inputs,
                              n_inputs + layer_k * max_hidden_per_layer)

        all_indices = jnp.arange(connectable_mask.shape[0])
        prior_mask = connectable_mask & (all_indices >= prior_start) & (all_indices < prior_end)
        return _select_random(prior_mask, max_connections, key)

    return connector


def lecun_uniform(
    key: jax.Array,
    shape: Tuple,
    dtype: jnp.dtype,
    fan_in: Int[Array, ''],
) -> Float[Array, 'max_connections']:
    """Lecun uniform: uniform(-limit, limit) where limit = sqrt(3 / fan_in)."""
    limit = jnp.sqrt(3.0 / jnp.maximum(fan_in, 1))
    return jax.random.uniform(key, shape, dtype, -limit, limit)


def make_weight_init_fn(
    weight_init: Callable = lecun_uniform,
) -> Callable:
    """Create a state_init_fn that initializes weights given a neuron with connectivity set.

    The returned function receives a NeuronState that already has incoming_ids and active_mask set.
    It initializes weights (and leaves other fields at their constructor defaults).

    Args:
        weight_init: Callable (key, shape, dtype, fan_in) -> Array.
            Defaults to lecun_uniform. fan_in is the number of active
            incoming connections, derived from the connection mask.

    Returns:
        state_init_fn(neuron_state, key) -> NeuronState
    """
    def state_init_fn(
        neuron_state: NeuronState,
        key: jax.Array,
    ) -> NeuronState:
        active_connection_mask = neuron_state.get_active_connection_mask()
        fan_in = active_connection_mask.sum()
        weights = weight_init(key, neuron_state.weights.shape, jnp.float32, fan_in)
        weights = jnp.where(active_connection_mask, weights, 0.0)
        return tree_replace(neuron_state, weights=weights)

    return state_init_fn


# ---------------------------------------------------------------------------
# MSE output error function
# ---------------------------------------------------------------------------

def make_mse_error_fn() -> Callable:
    """MSE derivative: 2 * (prediction - target) / n_outputs."""
    def compute_output_error(
        output_activations: Float[Array, 'n_outputs'],
        targets: Float[Array, 'n_outputs'],
    ) -> Float[Array, 'n_outputs']:
        n = output_activations.shape[0]
        return 2.0 * (output_activations - targets) / n

    return compute_output_error


# ---------------------------------------------------------------------------
# Convenience: full backprop StateUpdateFunctions
# ---------------------------------------------------------------------------

def make_backprop_sgd_update_functions(
    connectivity_init_fn: Callable,
    learning_rate: float = 0.01,
    activation_fn: Callable = jax.nn.relu,
    error_fn: Callable = make_mse_error_fn(),
):
    """Create a full StateUpdateFunctions for standard backprop + SGD.

    Builds weighted-sum forward, backprop error propagation, SGD weight
    updates, and linear output neurons. Output weights are zero-initialized.

    Args:
        connectivity_init_fn: Determines which neurons each new neuron
            connects from (e.g. ``make_prior_layer_connector``).
        learning_rate: SGD learning rate.
        activation_fn: Hidden-layer activation (default ReLU).
        error_fn: Output error function (activations, targets) -> errors.
            Defaults to MSE derivative.

    Returns:
        A ``StateUpdateFunctions`` instance ready for ``Network``.
    """
    identity = lambda x: x
    return StateUpdateFunctions(
        forward_fn=make_weighted_sum_forward_fn(activation_fn),
        backward_signal_fn=make_backprop_error_signal_fn(),
        neuron_update_fn=make_sgd_update_fn(learning_rate, activation_fn),
        structure_update_fn=noop_structure_update_fn,
        connectivity_init_fn=connectivity_init_fn,
        state_init_fn=make_weight_init_fn(),
        compute_output_error_fn=error_fn,
        output_forward_fn=make_weighted_sum_forward_fn(identity),
        output_backward_signal_fn=make_backprop_error_signal_fn(jax.grad(identity)),
        output_neuron_update_fn=make_sgd_update_fn(learning_rate, identity),
        output_state_init_fn=make_weight_init_fn(
            lambda key, shape, dtype, fan_in: jnp.zeros(shape, dtype)
        ),
    )
