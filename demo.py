"""Demo: Standard MLP with SGD on synthetic regression using plastax.

Task: y = [sin(x1 + x2), cos(x1 - x2)], 2 inputs, 2 outputs.
Uses backprop/SGD functions from plastax.standard.
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from plastax import (
    BackpropNeuronState,
    Network,
    StructureUpdateState,
    make_backprop_sgd_update_functions,
    make_prior_layer_connector,
)


LOG_INTERVAL = 500


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLP on sin/cos regression")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=50_000)
    return parser.parse_args()


class TrainState(eqx.Module):
    network: Network
    rng: jax.Array
    step: jax.Array


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def build_network(args: argparse.Namespace) -> TrainState:
    """Build and initialize a standard MLP."""
    n_inputs, n_outputs = 2, 2
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    max_conn = max(n_inputs, hidden_dim)

    # Define neuron types. BackpropNeuronState adds pre_activation and
    # incoming_activations fields needed by the standard forward/backward fns.
    # max_connections sets the maximum number of incoming connection each neuron can have.
    class HiddenNeuron(BackpropNeuronState):
        def __init__(self):
            super().__init__(max_connections=max_conn)

    class OutputNeuron(BackpropNeuronState):
        def __init__(self):
            super().__init__(max_connections=hidden_dim)

    # StateUpdateFunctions bundles all per-step behaviour: forward pass, error
    # propagation, weight updates, structure changes, and weight initialization.
    # make_backprop_sgd_update_functions wires up standard backprop + SGD with
    # sensible defaults (ReLU hidden, linear output, MSE, lecun_uniform init).
    # The connector determines how new neurons wire up. prior_layer_connector
    # connects each neuron to randomly-selected neurons in the layer before it.
    connector = make_prior_layer_connector(n_inputs, hidden_dim)
    fns = make_backprop_sgd_update_functions(connector, learning_rate=args.learning_rate)

    # Create the network. No neurons exist yet — just pre-allocated arrays.
    # max_generate_per_step=0 disables runtime neurogenesis.
    network = Network(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        max_hidden_per_layer=hidden_dim,
        max_layers=num_layers,
        max_generate_per_step=0,
        auto_connect_to_output=False,
        hidden_neuron_cls=HiddenNeuron,
        output_neuron_cls=OutputNeuron,
        state_update_fns=fns,
        structure_state=StructureUpdateState(),
    )

    # Populate hidden layers. add_layer uses connectivity_init_fn to wire each
    # neuron and state_init_fn to initialize its weights.
    key = random.PRNGKey(args.seed)
    for _ in range(num_layers):
        key, layer_key = random.split(key)
        network = network.add_layer(n_units=hidden_dim, key=layer_key)

    # Wire the last hidden layer to the output neurons.
    key, out_key = random.split(key)
    network = network.connect_to_output(network.get_units_in_layer(-1), out_key)

    return TrainState(network=network, rng=key, step=jnp.array(0))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def sample_data(key):
    x = random.uniform(key, (2,), minval=-jnp.pi, maxval=jnp.pi)
    y = jnp.array([jnp.sin(x[0] + x[1]), jnp.cos(x[0] - x[1])])
    return x, y


def train_step(state: TrainState, _):
    key, data_key, step_key = random.split(state.rng, 3)
    x, y = sample_data(data_key)
    # network.step runs forward, error computation, backward, weight update,
    # and structure update in one call.
    network = state.network.step(x, y, step_key)
    # Output neurons store their error signal; squaring gives per-unit loss.
    loss = jnp.mean(network.output_states.error_signal ** 2)
    return TrainState(network=network, rng=key, step=state.step + 1), loss


@jax.jit
def train_block(state: TrainState):
    return jax.lax.scan(train_step, state, length=LOG_INTERVAL)


def train(state: TrainState, num_steps: int) -> TrainState:
    for _ in range(num_steps // LOG_INTERVAL):
        state, losses = train_block(state)
        print(f"Step {int(state.step):>6d} | Loss: {float(losses.mean()):.6f}")
    return state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    state = build_network(args)

    print(f"Network: {args.num_layers} layer(s), {args.hidden_dim} hidden, lr={args.learning_rate}")
    print(f"Task: y = [sin(x1+x2), cos(x1-x2)]")

    train(state, args.num_steps)


if __name__ == "__main__":
    main()
