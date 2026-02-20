"""Shared fixtures and helpers for plastax tests."""

import jax
import jax.numpy as jnp
import pytest

from plastax import (
    CONNECTION_PADDING,
    BackpropNeuronState,
    Network,
    StateUpdateFunctions,
    StructureUpdateState,
    tree_replace,
)
from plastax.standard import (
    make_weighted_sum_forward_fn,
    noop_structure_update_fn,
)


# ---------------------------------------------------------------------------
# Neuron classes (small, hand-computable)
# ---------------------------------------------------------------------------

class SmallHiddenNeuron(BackpropNeuronState):
    """4 connection slots — enough for hand-computable tests."""
    def __init__(self):
        super().__init__(max_connections=4)


class SmallOutputNeuron(BackpropNeuronState):
    """4 connection slots for output neurons."""
    def __init__(self):
        super().__init__(max_connections=4)


# ---------------------------------------------------------------------------
# Deterministic init functions
# ---------------------------------------------------------------------------

def constant_weight_init(value=1.0):
    """Return a state_init_fn that sets all active weights to *value*."""
    def state_init_fn(neuron_state, key):
        weights = jnp.where(neuron_state.get_active_connection_mask(), value, 0.0)
        return tree_replace(neuron_state, weights=weights)
    return state_init_fn


def deterministic_connector(connectable_mask, index, max_connections, key):
    """Connect to all connectable neurons in index order (no randomness)."""
    indices = jnp.arange(connectable_mask.shape[0])
    sort_key = jnp.where(connectable_mask, 0, 1)
    sorted_indices = indices[jnp.argsort(sort_key, stable=True)][:max_connections]
    active = connectable_mask[sorted_indices]
    return jnp.where(active, sorted_indices, CONNECTION_PADDING)


def noop_backward_signal_fn(neuron_state, neuron_index, next_layer_states):
    """Always return zero error."""
    return jnp.array(0.0)


# ---------------------------------------------------------------------------
# StateUpdateFunctions builders
# ---------------------------------------------------------------------------

_identity_forward_fn = make_weighted_sum_forward_fn(lambda x: x)


def make_forward_only_fns(
    connectivity_init_fn=deterministic_connector,
    state_init_fn=None,
    output_state_init_fn=None,
    structure_update_fn=noop_structure_update_fn,
    forward_fn=None,
    output_forward_fn=None,
):
    """Build StateUpdateFunctions for forward-only tests (no learning)."""
    if state_init_fn is None:
        state_init_fn = constant_weight_init(1.0)
    if forward_fn is None:
        forward_fn = _identity_forward_fn
    if output_forward_fn is None:
        output_forward_fn = _identity_forward_fn
    return StateUpdateFunctions(
        forward_fn=forward_fn,
        backward_signal_fn=noop_backward_signal_fn,
        neuron_update_fn=lambda s: s,
        structure_update_fn=structure_update_fn,
        connectivity_init_fn=connectivity_init_fn,
        state_init_fn=state_init_fn,
        compute_output_error_fn=lambda preds, targets: jnp.zeros_like(preds),
        output_forward_fn=output_forward_fn,
        output_neuron_update_fn=lambda s: s,
        output_state_init_fn=output_state_init_fn if output_state_init_fn is not None else state_init_fn,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def forward_output(network, inputs, key):
    """Run a full step with dummy targets and return output activation values.

    Uses the public step() API. With make_forward_only_fns the backward and
    structure passes are noops, so weights and structure are unchanged.
    """
    targets = jnp.zeros(network.n_outputs)
    net = network.step(jnp.asarray(inputs, dtype=jnp.float32), targets, key)
    return net.output_states.activation_value


def extract_weight_matrix(neuron_states, n_sources, n_neurons):
    """Reconstruct a dense (n_neurons, n_sources) weight matrix from sparse storage.

    Uses incoming_ids/weights/active_connection_mask to scatter weights
    into a dense matrix.  Only the first n_neurons rows are used.
    """
    W = jnp.zeros((n_neurons, n_sources))
    ids = neuron_states.incoming_ids[:n_neurons]
    weights = neuron_states.weights[:n_neurons]
    mask = neuron_states.get_active_connection_mask()[:n_neurons]
    for i in range(n_neurons):
        for s in range(ids.shape[1]):
            if mask[i, s]:
                W = W.at[i, int(ids[i, s])].set(float(weights[i, s]))
    return W


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_net():
    """2 inputs, 1 output, max 4 hidden per layer, 2 layers.

    Identity activation everywhere, constant_weight_init(1.0).
    All hidden neurons start inactive.

    Index space:
        Inputs:         [0, 1]
        Hidden layer 0: abs [2, 3, 4, 5],  hidden-rel [0, 1, 2, 3]
        Hidden layer 1: abs [6, 7, 8, 9],  hidden-rel [4, 5, 6, 7]
        Output:         abs [10],           output-rel [0]
    """
    fns = make_forward_only_fns()
    return Network(
        n_inputs=2,
        n_outputs=1,
        max_hidden_per_layer=4,
        max_layers=2,
        hidden_neuron_cls=SmallHiddenNeuron,
        output_neuron_cls=SmallOutputNeuron,
        state_update_fns=fns,
        structure_state=StructureUpdateState(),
    )


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)
