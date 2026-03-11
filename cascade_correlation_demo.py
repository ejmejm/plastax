"""Cascade-Correlation on the two-spirals problem using plastax.

Implements the Cascade-Correlation Learning Architecture (Fahlman & Lebiere, 1990)
with the Quickprop optimizer. The network starts with inputs connected directly to
outputs, then grows by adding one hidden unit at a time — each trained to maximally
correlate with the residual error.

Architecture:
  - Hidden units: tanh activation, each in its own layer (cascade structure)
  - Output: linear (no activation), MSE loss, targets {-1, +1}
  - Optimizer: Quickprop (parabolic step with max growth factor mu=1.75)
  - Training: batch mode over all 194 patterns per epoch

Task: classify 194 points from two interleaved spirals (97 per class).
Typically solves with ~29 hidden units.
"""

import math
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from plastax import (
    BackpropNeuronState,
    Network,
    StateUpdateFunctions,
    StructureUpdateState,
    make_backprop_error_signal_fn,
    make_mse_error_fn,
    make_weight_init_fn,
    make_weighted_sum_forward_fn,
    noop_structure_update_fn,
    tree_replace,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_INPUTS = 2
N_OUTPUTS = 1
MAX_HIDDEN = 30
MAX_CONN = N_INPUTS + MAX_HIDDEN  # each unit can connect to all prior + inputs
N_CANDIDATES = 8

# Quickprop
QP_LR = 0.5
QP_MU = 1.75  # max growth factor

# Training
OUTPUT_MAX_EPOCHS = 2000
OUTPUT_PATIENCE = 100
CANDIDATE_MAX_EPOCHS = 1000
CANDIDATE_PATIENCE = 50


# ---------------------------------------------------------------------------
# Data: two spirals
# ---------------------------------------------------------------------------

def generate_spirals() -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Generate the classic 194-point two-spirals dataset.

    Each spiral has 97 points. Returns X:(194,2) and Y:(194,1).
    X values are centered at 0 in roughly [-1, 1].
    Y is +1.0 for spiral A, -1.0 for spiral B (linear output).
    """
    xs, ys = [], []
    for spiral_num in [1, -1]:
        for i in range(97):
            phi = i / 16 * math.pi
            r = 6.5 * (104 - i) / 104
            x = r * math.cos(phi) * spiral_num / 6.5
            y = r * math.sin(phi) * spiral_num / 6.5
            xs.append([x, y])
            ys.append([1.0 if spiral_num == 1 else -1.0])
    return jnp.array(xs), jnp.array(ys)


# ---------------------------------------------------------------------------
# Quickprop optimizer
# ---------------------------------------------------------------------------

class QuickpropState(eqx.Module):
    """Per-weight optimizer state for Quickprop."""
    prev_gradient: jax.Array
    prev_delta: jax.Array


def quickprop_step(
    weights: jax.Array,
    gradient: jax.Array,
    qp_state: QuickpropState,
    lr: float = QP_LR,
    mu: float = QP_MU,
    mask: jax.Array | None = None,
) -> Tuple[jax.Array, QuickpropState]:
    """One Quickprop weight update step.

    Args:
        weights: Current weight values.
        gradient: Current gradient (of loss for descent, of -S for ascent).
        qp_state: Previous gradient and delta.
        lr: Learning rate for gradient descent component.
        mu: Maximum growth factor.
        mask: Optional boolean mask — only update where True.

    Returns (new_weights, new_qp_state).
    """
    prev_grad = qp_state.prev_gradient
    prev_delta = qp_state.prev_delta

    # Quickprop parabolic step
    denom = prev_grad - gradient
    qp_delta = jnp.where(denom != 0, prev_delta * gradient / denom, 0.0)

    # Clip to max growth factor
    max_step = mu * jnp.abs(prev_delta)
    qp_delta = jnp.clip(qp_delta, -max_step, max_step)

    # First step: pure gradient descent; subsequent: parabolic + small epsilon nudge
    gd_delta = -lr * gradient
    epsilon_nudge = -0.01 * gradient
    is_first = jnp.all(prev_delta == 0)
    delta = jnp.where(is_first, gd_delta, qp_delta + epsilon_nudge)

    if mask is not None:
        delta = jnp.where(mask, delta, 0.0)

    new_weights = weights + delta
    new_qp_state = QuickpropState(prev_gradient=gradient, prev_delta=delta)
    return new_weights, new_qp_state


# ---------------------------------------------------------------------------
# Network setup
# ---------------------------------------------------------------------------

def noop_neuron_update(neuron_state):
    """No-op neuron update — weights stay frozen."""
    return neuron_state


def build_network() -> Network:
    """Build initial CasCor network: inputs → outputs only (no hidden units yet).

    Uses sigmoid activation for both hidden and output neurons.
    All weight updates are done externally via Quickprop, so both
    neuron_update_fn and output_neuron_update_fn are no-ops.
    """
    class HiddenNeuron(BackpropNeuronState):
        def __init__(self):
            super().__init__(max_connections=MAX_CONN)

    class OutputNeuron(BackpropNeuronState):
        def __init__(self):
            super().__init__(max_connections=MAX_CONN)

    # A dummy connector — we use add_unit with explicit incoming_ids,
    # so connectivity_init_fn is never called during normal operation.
    def dummy_connector(connectable_mask, index, max_connections, key):
        return jnp.zeros(max_connections, dtype=jnp.int32), jnp.zeros(max_connections, dtype=bool)

    fns = StateUpdateFunctions(
        forward_fn=make_weighted_sum_forward_fn(jnp.tanh),
        backward_signal_fn=make_backprop_error_signal_fn(),
        neuron_update_fn=noop_neuron_update,
        structure_update_fn=noop_structure_update_fn,
        connectivity_init_fn=dummy_connector,
        state_init_fn=make_weight_init_fn(),
        compute_output_error_fn=make_mse_error_fn(),
        output_forward_fn=make_weighted_sum_forward_fn(lambda x: x),
        output_neuron_update_fn=noop_neuron_update,
        output_state_init_fn=make_weight_init_fn(
            lambda key, shape, dtype, fan_in: jnp.zeros(shape, dtype)
        ),
    )

    # auto_connect_to_output=True wires outputs to inputs with zero weights.
    # This is the CasCor starting point: a direct input→output network.
    return Network(
        n_inputs=N_INPUTS,
        n_outputs=N_OUTPUTS,
        max_hidden_per_layer=1,
        max_layers=MAX_HIDDEN,
        max_generate_per_step=0,
        auto_connect_to_output=True,
        hidden_neuron_cls=HiddenNeuron,
        output_neuron_cls=OutputNeuron,
        state_update_fns=fns,
        structure_state=StructureUpdateState(),
    )


# ---------------------------------------------------------------------------
# Batch forward pass
# ---------------------------------------------------------------------------

def batch_forward(
    network: Network,
    X: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward all patterns through the network.

    Implements the forward pass manually so we can vmap over the dataset.
    Uses the network's stored weights, incoming_ids, and active masks.

    Args:
        network: The plastax Network.
        X: Input array of shape (n_patterns, n_inputs).

    Returns:
        all_activations: (n_patterns, n_inputs + total_hidden) — inputs + hidden acts.
        output_activations: (n_patterns, n_outputs).
    """
    n_patterns = X.shape[0]
    hidden = network.hidden_states
    output = network.output_states

    # Start with input activations, append zeros for hidden units
    all_acts = jnp.concatenate([X, jnp.zeros((n_patterns, network.total_hidden))], axis=1)

    # Process each hidden layer (Python loop unrolled at trace time)
    for layer_k in range(network.max_layers):
        start, end = network.layer_boundaries[layer_k]
        layer_ids = hidden.incoming_ids[start:end]          # (layer_size, max_conn)
        layer_weights = hidden.weights[start:end]            # (layer_size, max_conn)
        layer_conn_mask = hidden.active_connection_mask[start:end]  # (layer_size, max_conn)
        layer_active = hidden.active_mask[start:end]         # (layer_size,)

        # Gather incoming activations: (n_patterns, layer_size, max_conn)
        incoming = all_acts[:, layer_ids]

        # Weighted sum + tanh: (n_patterns, layer_size)
        z = (incoming * layer_weights[None, :, :] * layer_conn_mask[None, :, :]).sum(axis=-1)
        acts = jnp.tanh(z) * layer_active[None, :]

        # Place into all_activations at the right position
        all_acts = all_acts.at[:, N_INPUTS + start:N_INPUTS + end].set(acts)

    # Output layer
    out_ids = output.incoming_ids       # (n_outputs, max_out_conn)
    out_weights = output.weights        # (n_outputs, max_out_conn)
    out_conn_mask = output.active_connection_mask  # (n_outputs, max_out_conn)

    out_incoming = all_acts[:, out_ids]  # (n_patterns, n_outputs, max_out_conn)
    out_z = (out_incoming * out_weights[None, :, :] * out_conn_mask[None, :, :]).sum(axis=-1)
    output_acts = out_z  # linear output (no activation)

    return all_acts, output_acts


@jax.jit
def _evaluate_jit(network: Network, X: jnp.ndarray, Y: jnp.ndarray):
    """Batch evaluation (JIT-compiled)."""
    _, output_acts = batch_forward(network, X)
    loss = jnp.mean((output_acts - Y) ** 2)
    predictions = jnp.where(output_acts > 0.0, 1.0, -1.0)
    n_correct = jnp.sum(predictions == Y)
    return loss, n_correct


def evaluate(network: Network, X: jnp.ndarray, Y: jnp.ndarray) -> Tuple[float, int]:
    """Compute MSE loss and count correct classifications."""
    loss, n_correct = _evaluate_jit(network, X, Y)
    return float(loss), int(n_correct)


# ---------------------------------------------------------------------------
# Phase 1: Output weight training (Quickprop)
# ---------------------------------------------------------------------------

@jax.jit
def _output_train_step(network, output_weights, qp_state, conn_mask, X, Y):
    """One output-weight Quickprop epoch (JIT-compiled).

    Returns (new_weights, new_qp_state, loss_before_update).
    """
    def loss_fn(ow):
        net = eqx.tree_at(lambda n: n.output_states.weights, network, ow)
        _, output_acts = batch_forward(net, X)
        return jnp.mean((output_acts - Y) ** 2)

    loss, g = jax.value_and_grad(loss_fn)(output_weights)
    new_weights, new_qp = quickprop_step(output_weights, g, qp_state, mask=conn_mask)
    new_weights = jnp.where(conn_mask, new_weights, 0.0)
    return new_weights, new_qp, loss


def train_outputs(
    network: Network,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    max_epochs: int = OUTPUT_MAX_EPOCHS,
    patience: int = OUTPUT_PATIENCE,
) -> Network:
    """Train output weights via batch Quickprop, hidden weights frozen."""
    output_weights = network.output_states.weights
    conn_mask = network.output_states.active_connection_mask
    qp_state = QuickpropState(
        prev_gradient=jnp.zeros_like(output_weights),
        prev_delta=jnp.zeros_like(output_weights),
    )

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(max_epochs):
        output_weights, qp_state, loss = _output_train_step(
            network, output_weights, qp_state, conn_mask, X, Y)

        current_loss = float(loss)
        if current_loss < best_loss - 1e-6:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    network = eqx.tree_at(lambda n: n.output_states.weights, network, output_weights)
    return network


# ---------------------------------------------------------------------------
# Phase 2: Candidate training (correlation maximization)
# ---------------------------------------------------------------------------

def compute_S(
    candidate_weights: jnp.ndarray,
    conn_mask: jnp.ndarray,
    source_acts: jnp.ndarray,
    residual_errors: jnp.ndarray,
) -> jnp.ndarray:
    """Compute the correlation metric S for a single candidate.

    S = Σ_o |Σ_p (V_p - V̄)(E_{p,o} - Ē_o)|

    Args:
        candidate_weights: (max_conn,) weight vector.
        conn_mask: (max_conn,) boolean mask of active connections.
        source_acts: (n_patterns, max_conn) activations of sources for all patterns.
        residual_errors: (n_patterns, n_outputs) residual errors.
    """
    z = (source_acts * candidate_weights[None, :] * conn_mask[None, :]).sum(axis=-1)  # (n_patterns,)
    V = jnp.tanh(z)
    V_bar = jnp.mean(V)
    E_bar = jnp.mean(residual_errors, axis=0)  # (n_outputs,)

    # Covariance per output
    cov = jnp.sum((V[:, None] - V_bar) * (residual_errors - E_bar[None, :]), axis=0)  # (n_outputs,)
    return jnp.sum(jnp.abs(cov))


@jax.jit
def _candidate_train_step(candidate_weights, qp_states, conn_mask, source_acts, residuals):
    """One candidate-pool Quickprop epoch (JIT-compiled).

    Computes gradients of S, applies Quickprop ascent, returns S at new weights.
    """
    grads = jax.vmap(jax.grad(compute_S), in_axes=(0, None, None, None))(
        candidate_weights, conn_mask, source_acts, residuals)

    neg_grads = -grads
    new_weights, new_qp = jax.vmap(
        quickprop_step, in_axes=(0, 0, QuickpropState(0, 0), None, None, None)
    )(candidate_weights, neg_grads, qp_states, QP_LR, QP_MU, conn_mask)

    new_weights = jnp.where(conn_mask[None, :], new_weights, 0.0)

    S_vals = jax.vmap(compute_S, in_axes=(0, None, None, None))(
        new_weights, conn_mask, source_acts, residuals)

    return new_weights, new_qp, S_vals


def train_candidates(
    network: Network,
    X: jnp.ndarray,
    Y: jnp.ndarray,
    key: jax.Array,
    n_installed: int,
    n_candidates: int = N_CANDIDATES,
    max_epochs: int = CANDIDATE_MAX_EPOCHS,
    patience: int = CANDIDATE_PATIENCE,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Train a pool of candidate units to maximize correlation S.

    Each candidate connects to all inputs + all installed hidden units.
    Uses Quickprop gradient ascent (negate gradient for maximization).

    Returns (best_weights, best_incoming_ids) for the winning candidate.
    """
    n_sources = N_INPUTS + n_installed

    # Get source activations and residual errors (JIT-compiled forward pass)
    all_acts, output_acts = jax.jit(batch_forward)(network, X)
    residual_errors = output_acts - Y

    # Pad source activations to max_conn width
    n_pad = MAX_CONN - all_acts.shape[1]
    if n_pad > 0:
        source_acts_padded = jnp.concatenate(
            [all_acts, jnp.zeros((X.shape[0], n_pad))], axis=1)
    else:
        source_acts_padded = all_acts[:, :MAX_CONN]

    # Connection mask: first n_sources slots active
    conn_mask = jnp.arange(MAX_CONN) < n_sources

    # Initialize candidate weights randomly
    key, init_key = random.split(key)
    candidate_weights = random.uniform(
        init_key, (n_candidates, MAX_CONN), minval=-0.5, maxval=0.5)
    candidate_weights = jnp.where(conn_mask[None, :], candidate_weights, 0.0)

    qp_states = QuickpropState(
        prev_gradient=jnp.zeros((n_candidates, MAX_CONN)),
        prev_delta=jnp.zeros((n_candidates, MAX_CONN)),
    )

    best_S_overall = jnp.zeros(n_candidates)
    best_weights_overall = candidate_weights.copy()

    best_S_val = -jnp.inf
    patience_counter = 0

    for epoch in range(max_epochs):
        candidate_weights, qp_states, S_vals = _candidate_train_step(
            candidate_weights, qp_states, conn_mask, source_acts_padded, residual_errors)

        # Track best weights per candidate
        improved = S_vals > best_S_overall
        best_S_overall = jnp.where(improved, S_vals, best_S_overall)
        best_weights_overall = jnp.where(
            improved[:, None], candidate_weights, best_weights_overall)

        # Patience on best S across all candidates
        current_best = float(jnp.max(S_vals))
        if current_best > float(best_S_val) + 1e-6:
            best_S_val = current_best
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    winner_idx = jnp.argmax(best_S_overall)
    best_weights = best_weights_overall[winner_idx]
    print(f"  Candidate training: best S={float(best_S_overall[winner_idx]):.4f} "
          f"(epoch {epoch+1}/{max_epochs})")

    # Build incoming_ids for the winner
    incoming_ids = jnp.full(MAX_CONN, -1, dtype=jnp.int32)
    incoming_ids = incoming_ids.at[:n_sources].set(jnp.arange(n_sources, dtype=jnp.int32))

    return best_weights, incoming_ids


# ---------------------------------------------------------------------------
# Candidate installation
# ---------------------------------------------------------------------------

def install_candidate(
    network: Network,
    candidate_weights: jnp.ndarray,
    incoming_ids: jnp.ndarray,
    n_installed: int,
    key: jax.Array,
) -> Network:
    """Install a trained candidate as a permanent hidden unit.

    1. Adds the unit to the network via add_unit (which infers the correct layer).
    2. Overwrites the state_init_fn weights with the pre-trained candidate weights.
    3. Connects the new unit to the output layer.

    NOTE: add_unit always runs state_init_fn, so we immediately overwrite
    the weights. This is awkward — a bypass option would be cleaner.
    """
    add_key, connect_key = random.split(key)

    # Add the unit. It will be placed in layer n_installed.
    network, success = network.add_unit(incoming_ids, add_key)

    # The new unit is at hidden_states index n_installed
    # (since max_hidden_per_layer=1, layer k starts at index k).
    hidden_idx = n_installed

    # Overwrite weights with trained candidate weights
    new_weights = network.hidden_states.weights.at[hidden_idx].set(candidate_weights)
    network = eqx.tree_at(
        lambda n: n.hidden_states.weights,
        network,
        new_weights,
    )

    # Connect to output. The new unit's absolute index is n_inputs + hidden_idx.
    abs_idx = jnp.array([N_INPUTS + hidden_idx])
    network = network.connect_to_output(abs_idx, connect_key)

    return network


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Cascade-Correlation on the Two-Spirals Problem")
    print("=" * 50)

    # Generate data
    X, Y = generate_spirals()
    print(f"Dataset: {X.shape[0]} points, {N_INPUTS} inputs, {N_OUTPUTS} output")

    # Build initial network (inputs → outputs only)
    network = build_network()

    # Initialize output weights randomly (auto_connect_to_output sets them to zero,
    # but zero weights give near-zero gradients for symmetric data like spirals).
    key = random.PRNGKey(42)
    key, init_key = random.split(key)
    init_weights = random.normal(init_key, network.output_states.weights.shape) * 0.5
    init_weights = jnp.where(network.output_states.active_connection_mask, init_weights, 0.0)
    network = eqx.tree_at(lambda n: n.output_states.weights, network, init_weights)

    # Initial evaluation
    loss, n_correct = evaluate(network, X, Y)
    print(f"\nInitial: loss={loss:.6f}, accuracy={n_correct}/194")

    for step in range(MAX_HIDDEN):
        print(f"\n--- Cascade step {step} ---")

        # Phase 1: Train output weights
        network = train_outputs(network, X, Y)
        loss, n_correct = evaluate(network, X, Y)
        print(f"After output training: loss={loss:.6f}, accuracy={n_correct}/194")

        if n_correct == 194:
            print("\nSolved! All 194 patterns classified correctly.")
            break

        # Phase 2: Train and install candidate
        key, cand_key, install_key = random.split(key, 3)
        candidate_weights, incoming_ids = train_candidates(
            network, X, Y, cand_key, n_installed=step)

        network = install_candidate(
            network, candidate_weights, incoming_ids, n_installed=step, key=install_key)

        loss, n_correct = evaluate(network, X, Y)
        print(f"After installation:   loss={loss:.6f}, accuracy={n_correct}/194")
    else:
        print(f"\nReached max hidden units ({MAX_HIDDEN}) without solving.")
        loss, n_correct = evaluate(network, X, Y)
        print(f"Final: loss={loss:.6f}, accuracy={n_correct}/194")

    # Library compatibility report
    print("\n" + "=" * 50)
    print("LIBRARY COMPATIBILITY NOTES")
    print("=" * 50)
    notes = [
        ("No batch processing",
         "network.step() processes one sample at a time. Quickprop and\n"
         "  correlation maximization require full-dataset batch operations,\n"
         "  so the training loop was implemented externally."),
        ("No selective weight freezing",
         "To freeze hidden weights during output training, a custom noop\n"
         "  neuron_update_fn was needed. The library has no built-in\n"
         "  freeze mechanism."),
        ("Skip-connection backward pass",
         "The layer-by-layer backward pass only propagates error from the\n"
         "  immediate next layer. With cascade skip connections, hidden\n"
         "  error signals would be incomplete. This doesn't affect CasCor\n"
         "  (hidden weights are always frozen) but would break standard\n"
         "  backprop with skip connections."),
        ("Weight override after add_unit",
         "add_unit() always runs state_init_fn, so pre-trained candidate\n"
         "  weights must be overwritten immediately via eqx.tree_at.\n"
         "  An option to pass pre-initialized state would be cleaner."),
        ("Layer-per-unit architecture",
         "max_hidden_per_layer=1 correctly models the cascade structure\n"
         "  (each unit depends on all prior units) but pre-allocates one\n"
         "  full neuron slot per layer. A flat neuron pool would be more\n"
         "  natural for cascade architectures."),
        ("Correlation objective outside library",
         "The library's training pipeline assumes error minimization.\n"
         "  Correlation maximization is a fundamentally different objective\n"
         "  that must be implemented entirely outside the library."),
    ]
    for i, (title, detail) in enumerate(notes, 1):
        print(f"\n{i}. {title}:")
        print(f"  {detail}")


if __name__ == "__main__":
    main()
