"""Continual Backprop demo on the bit-flipping regression task.

Compares standard backprop (which loses plasticity over time) against
Continual Backprop (Dohare et al. 2021) which maintains plasticity by
continuously replacing low-utility neurons.

Task: 20 binary inputs → scalar target from a fixed LTU network.
      15 "flipping bits" toggle one-at-a-time every T steps (default 10,000),
      5 random i.i.d. bits change each step.

Usage:
    python continual_backprop_demo.py
    python continual_backprop_demo.py --num_steps 500000 --seed 42
"""

import argparse
from typing import Callable, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import random
from jaxtyping import Array, Bool, Float, Int

from plastax import (
    BackpropNeuronState,
    Network,
    StateUpdateFunctions,
    StructureUpdateState,
    make_backprop_error_signal_fn,
    make_mse_error_fn,
    make_prior_layer_connector,
    make_sgd_update_fn,
    make_weight_init_fn,
    make_weighted_sum_forward_fn,
    noop_structure_update_fn,
    tree_replace,
    lecun_uniform,
)


# ── Constants ──────────────────────────────────────────────────────────

N_INPUTS = 20
N_FLIPPING = 15
N_IID = 5
N_OUTPUTS = 1
TARGET_HIDDEN = 100
LOG_INTERVAL = 1000


# ── Argument parsing ──────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continual Backprop on bit-flipping regression")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_steps", type=int, default=500_000)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=10,
                        help="Learner hidden layer width")
    parser.add_argument("--flip_period", type=int, default=10_000,
                        help="Steps between bit flips")
    parser.add_argument("--rho", type=float, default=1e-3,
                        help="CBP replacement rate")
    parser.add_argument("--eta", type=float, default=0.99,
                        help="CBP utility EMA decay rate")
    parser.add_argument("--maturity", type=int, default=100,
                        help="CBP maturity threshold (steps before eligible)")
    return parser.parse_args()


# ── Target network (fixed LTU) ───────────────────────────────────────

class TargetNetwork(eqx.Module):
    """Two-layer Linear Threshold Unit network with binary weights.

    Used as the fixed ground-truth function. The learner never sees
    this network's weights — only its outputs.
    """
    w1: Float[Array, 'target_hidden n_inputs']
    w2: Float[Array, '1 target_hidden']

    def __init__(self, key):
        k1, k2 = random.split(key)
        self.w1 = 2.0 * random.bernoulli(
            k1, shape=(TARGET_HIDDEN, N_INPUTS)).astype(jnp.float32) - 1.0
        self.w2 = 2.0 * random.bernoulli(
            k2, shape=(1, TARGET_HIDDEN)).astype(jnp.float32) - 1.0

    def __call__(self, x: Float[Array, 'n_inputs']) -> Float[Array, '']:
        h = (self.w1 @ x > 0).astype(jnp.float32)
        return (self.w2 @ h > 0).astype(jnp.float32).squeeze()


# ── Bit-flipping environment ─────────────────────────────────────────

class BitFlipState(eqx.Module):
    """Tracks the current state of the 15 flipping bits and step counter."""
    flipping_bits: Float[Array, 'n_flipping']
    step: Int[Array, '']


def make_generate_sample(
    flip_sequence: Int[Array, 'n_flips'],
    target: TargetNetwork,
    flip_period: int,
):
    """Create a function that generates one (x, y) sample and advances the
    bit-flipping environment.

    flip_sequence[i] is the index of the bit to flip at the i-th flip event.
    Flip events happen every flip_period steps (starting at step flip_period).
    """
    def generate_sample(
        bit_state: BitFlipState,
        data_key: jax.Array,
    ) -> Tuple[BitFlipState, Float[Array, 'n_inputs'], Float[Array, '']]:
        step = bit_state.step

        # Determine if a bit flip happens this step
        flip_event = (step > 0) & (step % flip_period == 0)
        flip_index = jnp.minimum(
            step // flip_period, flip_sequence.shape[0] - 1)
        which_bit = flip_sequence[flip_index]

        # Apply the flip (toggle the selected bit)
        old_val = bit_state.flipping_bits[which_bit]
        new_bits = jnp.where(
            flip_event,
            bit_state.flipping_bits.at[which_bit].set(1.0 - old_val),
            bit_state.flipping_bits,
        )

        # Generate 5 random i.i.d. bits
        iid_bits = random.bernoulli(
            data_key, shape=(N_IID,)).astype(jnp.float32)

        # Full input: 15 flipping + 5 random = 20 bits
        x = jnp.concatenate([new_bits, iid_bits])

        # Target output from the fixed LTU network
        y = target(x)

        new_state = BitFlipState(flipping_bits=new_bits, step=step + 1)
        return new_state, x, y

    return generate_sample


# ── CBP structure state and update ───────────────────────────────────

class CBPStructureState(StructureUpdateState):
    """Tracks per-neuron utility and age for Continual Backprop.

    Stored in network.structure_state, updated by the structure_update_fn
    at each step.
    """
    utility: Float[Array, 'hidden_dim']
    ages: Float[Array, 'hidden_dim']
    replacement_budget: Float[Array, '']


def make_cbp_structure_update_fn(
    n_inputs: int,
    eta: float = 0.99,
    rho: float = 1e-4,
    maturity_threshold: int = 100,
) -> Callable:
    """Create a structure_update_fn that implements Continual Backprop.

    Computes utility = |activation| × Σ|outgoing weights| as an EMA,
    and replaces the lowest-utility mature neuron when the accumulated
    replacement budget reaches 1.

    NOTE (library limitation): structure_update_fn receives no PRNG key,
    so replacement scheduling must be deterministic. We use a fractional
    budget accumulator instead of stochastic sampling. Fix: add a key
    parameter to the structure_update_fn signature — the key is already
    available in Network._structure_update().

    NOTE (library limitation): structure_update_fn doesn't know which
    layer it's processing. We must close over n_inputs to reconstruct
    absolute neuron indices. Fix: pass layer metadata (layer index,
    absolute start index, n_inputs) as additional arguments.
    """
    def cbp_structure_update_fn(
        layer_states: BackpropNeuronState,
        next_layer_states: BackpropNeuronState,
        structure_state: CBPStructureState,
    ) -> Tuple[CBPStructureState, Bool[Array, 'layer_size'], Int[Array, '']]:
        layer_size = layer_states.active_mask.shape[0]
        active = layer_states.active_mask

        # 1. Increment ages for active neurons
        new_ages = structure_state.ages + active.astype(jnp.float32)

        # 2. Compute outgoing weight sums for each hidden neuron.
        #    For neuron j in this layer, absolute index = n_inputs + j.
        #    Check next_layer_states.incoming_ids for connections pointing here.
        abs_indices = jnp.arange(layer_size) + n_inputs
        # matches[j, next_neuron, conn_slot]: does this connection come from j?
        matches = (
            (next_layer_states.incoming_ids[None, :, :] == abs_indices[:, None, None])
            & next_layer_states.active_connection_mask[None, :, :]
        )
        outgoing_weight_sum = (
            jnp.abs(next_layer_states.weights[None, :, :]) * matches
        ).sum(axis=(1, 2))

        # 3. Instantaneous utility: |activation| × outgoing weight sum
        instant_utility = jnp.abs(layer_states.activation_value) * outgoing_weight_sum

        # 4. EMA update
        new_utility = eta * structure_state.utility + (1 - eta) * instant_utility
        new_utility = jnp.where(active, new_utility, 0.0)

        # 5. Replacement decision
        eligible = active & (new_ages >= maturity_threshold)
        n_eligible = eligible.sum()
        new_budget = structure_state.replacement_budget + rho * n_eligible
        should_replace = new_budget >= 1.0

        # Select the lowest-utility eligible neuron
        masked_utility = jnp.where(eligible, new_utility, jnp.inf)
        worst_idx = jnp.argmin(masked_utility)

        prune_mask = jnp.zeros(layer_size, dtype=bool).at[worst_idx].set(
            should_replace & eligible.any())
        n_generate = prune_mask.sum().astype(jnp.int32)

        # Reset tracking for the pruned neuron and decrement budget
        new_ages = jnp.where(prune_mask, 0.0, new_ages)
        new_utility = jnp.where(prune_mask, 0.0, new_utility)
        final_budget = jnp.where(
            should_replace & eligible.any(),
            new_budget - 1.0,
            new_budget,
        )

        new_state = CBPStructureState(
            utility=new_utility,
            ages=new_ages,
            replacement_budget=final_budget,
        )
        return new_state, prune_mask, n_generate

    return cbp_structure_update_fn


# ── Output reconnection ──────────────────────────────────────────────

def make_ensure_output_connectivity(n_inputs: int, hidden_dim: int):
    """Create a function that enforces fully-connected output connectivity.

    After a neuron is pruned and regenerated, the output layer loses the
    connection to that neuron's slot. This function rebuilds the expected
    connectivity every step, transferring existing trained weights and
    setting new connections to zero weight.

    NOTE (library limitation): there is no automatic output reconnection
    after neuron replacement when auto_connect_to_output=False. Fix: after
    _generate_into_layer, detect previously-connected output slots that now
    point to newly active neurons and reactivate them. Or provide a
    post_generate_hook in StateUpdateFunctions.
    """
    expected_ids = jnp.arange(hidden_dim, dtype=jnp.int32) + n_inputs

    def ensure_output_connectivity(network: Network) -> Network:
        hidden_active = network.hidden_states.active_mask[:hidden_dim]
        output_states = network.output_states

        # For each expected connection, find the old weight (if any).
        # matches[out, old_slot, new_slot] = True if old slot had this ID
        matches = (
            (output_states.incoming_ids[:, :, None] == expected_ids[None, None, :])
            & output_states.active_connection_mask[:, :, None]
        )
        gathered_weights = (
            output_states.weights[:, :, None] * matches
        ).sum(axis=1)
        was_connected = matches.any(axis=1)
        new_weights = jnp.where(was_connected, gathered_weights, 0.0)

        # Set connectivity: slot i → hidden neuron i
        n_out = output_states.incoming_ids.shape[0]
        new_ids = jnp.tile(expected_ids, (n_out, 1))
        new_mask = jnp.tile(hidden_active, (n_out, 1))

        return tree_replace(
            network,
            output_states=tree_replace(
                output_states,
                incoming_ids=new_ids,
                active_connection_mask=new_mask,
                weights=new_weights,
            ),
        )

    return ensure_output_connectivity


# ── Network building ─────────────────────────────────────────────────

def build_network(
    hidden_dim: int,
    structure_update_fn: Callable,
    structure_state: StructureUpdateState,
    learning_rate: float,
    key: jax.Array,
) -> Network:
    """Build a N_INPUTS → hidden_dim (ReLU) → 1 (linear) network.

    Both experiments use the same architecture and initial weights — the
    only difference is the structure_update_fn (noop vs CBP).
    """
    max_conn = N_INPUTS  # hidden neurons connect to all 20 inputs

    class HiddenNeuron(BackpropNeuronState):
        def __init__(self):
            super().__init__(max_connections=max_conn)

    class OutputNeuron(BackpropNeuronState):
        def __init__(self):
            super().__init__(max_connections=hidden_dim)

    connector = make_prior_layer_connector(N_INPUTS, hidden_dim)
    identity = lambda x: x

    fns = StateUpdateFunctions(
        forward_fn=make_weighted_sum_forward_fn(jax.nn.relu),
        backward_signal_fn=make_backprop_error_signal_fn(),
        neuron_update_fn=make_sgd_update_fn(learning_rate, jax.nn.relu),
        structure_update_fn=structure_update_fn,
        connectivity_init_fn=connector,
        state_init_fn=make_weight_init_fn(),
        compute_output_error_fn=make_mse_error_fn(),
        output_forward_fn=make_weighted_sum_forward_fn(identity),
        output_backward_signal_fn=make_backprop_error_signal_fn(jax.grad(identity)),
        output_neuron_update_fn=make_sgd_update_fn(learning_rate, identity),
        output_state_init_fn=make_weight_init_fn(
            lambda key, shape, dtype, fan_in: jnp.zeros(shape, dtype)
        ),
    )

    network = Network(
        n_inputs=N_INPUTS,
        n_outputs=N_OUTPUTS,
        max_hidden_per_layer=hidden_dim,
        max_layers=1,
        max_generate_per_step=1,
        auto_connect_to_output=False,
        hidden_neuron_cls=HiddenNeuron,
        output_neuron_cls=OutputNeuron,
        state_update_fns=fns,
        structure_state=structure_state,
    )

    # Add hidden layer and wire to output
    key, layer_key = random.split(key)
    network = network.add_layer(
        n_units=hidden_dim, key=layer_key, connect_to_output=True)

    return network


# ── Training ─────────────────────────────────────────────────────────

class StandardTrainState(eqx.Module):
    network: Network
    bit_state: BitFlipState
    rng: jax.Array


class CBPTrainState(eqx.Module):
    network: Network
    bit_state: BitFlipState
    rng: jax.Array


def make_standard_train_step(generate_sample):
    """Create a train_step for standard backprop (no CBP replacement)."""

    def train_step(state: StandardTrainState, _):
        key, data_key, step_key = random.split(state.rng, 3)
        bit_state, x, y = generate_sample(state.bit_state, data_key)

        network = state.network.step(x, y.reshape(N_OUTPUTS), step_key)

        pred = network.output_states.activation_value[0]
        loss = (pred - y) ** 2

        return StandardTrainState(
            network=network, bit_state=bit_state, rng=key), loss

    return train_step


def make_cbp_train_step(generate_sample, ensure_connectivity):
    """Create a train_step for Continual Backprop (with replacement + reconnect)."""

    def train_step(state: CBPTrainState, _):
        key, data_key, step_key = random.split(state.rng, 3)
        bit_state, x, y = generate_sample(state.bit_state, data_key)

        network = state.network.step(x, y.reshape(N_OUTPUTS), step_key)

        # Reconnect output after potential neuron replacement
        network = ensure_connectivity(network)

        pred = network.output_states.activation_value[0]
        loss = (pred - y) ** 2

        return CBPTrainState(
            network=network, bit_state=bit_state, rng=key), loss

    return train_step


def run_experiment(name, initial_state, train_step_fn, num_steps):
    """Run an experiment using jax.lax.scan in blocks."""

    @jax.jit
    def train_block(state):
        return jax.lax.scan(train_step_fn, state, length=LOG_INTERVAL)

    state = initial_state
    all_losses = []
    n_blocks = num_steps // LOG_INTERVAL

    for i in range(n_blocks):
        state, losses = train_block(state)
        mean_loss = float(losses.mean())
        step = (i + 1) * LOG_INTERVAL
        if step % (LOG_INTERVAL * 10) == 0:
            print(f"  [{name}] Step {step:>7d} | Avg loss: {mean_loss:.6f}")
        all_losses.append(losses)

    return state, jnp.concatenate(all_losses)


# ── Plotting ─────────────────────────────────────────────────────────

def plot_results(
    losses_bp: Float[Array, 'n_steps'],
    losses_cbp: Float[Array, 'n_steps'],
    flip_period: int,
    window: int = 1000,
):
    """Plot running average squared error for both methods."""
    kernel = jnp.ones(window) / window
    smooth_bp = jnp.convolve(losses_bp, kernel, mode='valid')
    smooth_cbp = jnp.convolve(losses_cbp, kernel, mode='valid')
    steps = jnp.arange(window - 1, len(losses_bp))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(steps, smooth_bp, label='Standard Backprop', alpha=0.8)
    ax.plot(steps, smooth_cbp, label='Continual Backprop', alpha=0.8)

    # Mark bit-flip events
    for t in range(flip_period, len(losses_bp), flip_period):
        ax.axvline(t, color='gray', alpha=0.15, linestyle='--')

    ax.set_xlabel('Step')
    ax.set_ylabel('Running Average Squared Error')
    ax.set_title('Bit-Flipping Task: Loss of Plasticity')
    ax.legend()

    plt.tight_layout()
    plt.savefig('continual_backprop_results.png', dpi=150)
    print("Plot saved to continual_backprop_results.png")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    hidden_dim = args.hidden_dim
    flip_period = args.flip_period
    key = random.PRNGKey(args.seed)

    # Generate fixed target network and flip schedule
    key, target_key, flip_key, init_bits_key = random.split(key, 4)
    target = TargetNetwork(target_key)

    n_flips = args.num_steps // flip_period + 1
    flip_sequence = random.randint(flip_key, (n_flips,), 0, N_FLIPPING)

    generate_sample = make_generate_sample(flip_sequence, target, flip_period)
    ensure_connectivity = make_ensure_output_connectivity(N_INPUTS, hidden_dim)

    # Build networks — same initial weights, different structure_update_fn
    key, net_key = random.split(key)

    network_bp = build_network(
        hidden_dim=hidden_dim,
        structure_update_fn=noop_structure_update_fn,
        structure_state=StructureUpdateState(),
        learning_rate=args.learning_rate,
        key=net_key,
    )

    cbp_state = CBPStructureState(
        utility=jnp.zeros(hidden_dim),
        ages=jnp.zeros(hidden_dim),
        replacement_budget=jnp.array(0.0),
    )
    cbp_fn = make_cbp_structure_update_fn(
        n_inputs=N_INPUTS,
        eta=args.eta,
        rho=args.rho,
        maturity_threshold=args.maturity,
    )
    network_cbp = build_network(
        hidden_dim=hidden_dim,
        structure_update_fn=cbp_fn,
        structure_state=cbp_state,
        learning_rate=args.learning_rate,
        key=net_key,  # same key → same initial weights
    )

    # Shared initial environment state
    initial_bits = random.bernoulli(
        init_bits_key, shape=(N_FLIPPING,)).astype(jnp.float32)
    bit_state = BitFlipState(flipping_bits=initial_bits, step=jnp.array(0))

    key, run_key = random.split(key)

    state_bp = StandardTrainState(
        network=network_bp, bit_state=bit_state, rng=run_key)
    state_cbp = CBPTrainState(
        network=network_cbp, bit_state=bit_state, rng=run_key)

    train_step_bp = make_standard_train_step(generate_sample)
    train_step_cbp = make_cbp_train_step(generate_sample, ensure_connectivity)

    print(f"Bit-flipping task: {N_INPUTS} inputs, {hidden_dim} hidden, "
          f"{N_OUTPUTS} output")
    print(f"Target: {TARGET_HIDDEN}-unit LTU network")
    print(f"Flip period: {flip_period} steps, Total: {args.num_steps} steps")
    print(f"CBP: rho={args.rho}, eta={args.eta}, maturity={args.maturity}")
    print()

    # Run standard backprop
    print("Running Standard Backprop...")
    _, losses_bp = run_experiment(
        "BP", state_bp, train_step_bp, args.num_steps)

    # Run Continual Backprop
    print("\nRunning Continual Backprop...")
    _, losses_cbp = run_experiment(
        "CBP", state_cbp, train_step_cbp, args.num_steps)

    # Plot comparison
    plot_results(losses_bp, losses_cbp, flip_period)

    # Print library notes
    print("\n── Library Compatibility Notes ──")
    print(
        "1. auto_connect_to_output conflates constructor behavior (wiring\n"
        "   outputs to inputs) with runtime behavior (reconnecting generated\n"
        "   neurons to outputs). Fix: split into separate flags.\n"
    )
    print(
        "2. structure_update_fn receives no PRNG key for stochastic decisions.\n"
        "   Fix: add a key parameter — the key is already available in\n"
        "   Network._structure_update().\n"
    )
    print(
        "3. No automatic output reconnection after neuron replacement when\n"
        "   auto_connect_to_output=False. Fix: detect previously-connected\n"
        "   output slots pointing to newly active neurons and reactivate,\n"
        "   or add a post_generate_hook to StateUpdateFunctions.\n"
    )
    print(
        "4. structure_update_fn doesn't know which layer it's processing.\n"
        "   Must close over n_inputs and layer offset in a factory.\n"
        "   Fix: pass layer metadata (index, start, n_inputs) as arguments.\n"
    )


if __name__ == "__main__":
    main()
