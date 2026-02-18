"""Demo: Standard MLP with SGD on synthetic regression using plastax.

Task: y = [sin(x1 + x2), cos(x1 - x2)], 2 inputs, 2 outputs.
Uses default forward/backward/update functions from plastax.defaults.
"""

import argparse

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random

from plastax import (
    DefaultNeuronState,
    Network,
    StateUpdateFunctions,
    StructureUpdateState,
    default_structure_update_fn,
    make_default_backward_signal_fn,
    make_default_forward_fn,
    make_default_neuron_update_fn,
    make_default_output_error_fn,
    make_default_state_init_fn,
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


# ---------------------------------------------------------------------------
# TrainState
# ---------------------------------------------------------------------------

class TrainState(eqx.Module):
    network: Network
    rng: jax.Array
    step: jax.Array


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_experiment(args: argparse.Namespace) -> TrainState:
    """Build network and wire up a standard MLP."""
    n_inputs = 2
    n_outputs = 2
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    max_hidden_conn = max(n_inputs, hidden_dim)

    # Neuron classes with baked-in max_connections
    class HiddenNeuron(DefaultNeuronState):
        def __init__(self):
            super().__init__(max_connections=max_hidden_conn)

    class OutputNeuron(DefaultNeuronState):
        def __init__(self):
            super().__init__(max_connections=hidden_dim)

    # State update functions (ReLU hidden, linear output, SGD weight updates)
    lr = args.learning_rate
    identity = lambda x: x
    connector = make_prior_layer_connector(n_inputs, hidden_dim)

    fns = StateUpdateFunctions(
        forward_fn=make_default_forward_fn(jax.nn.relu),
        backward_signal_fn=make_default_backward_signal_fn(),
        neuron_update_fn=make_default_neuron_update_fn(lr, jax.nn.relu),
        structure_update_fn=default_structure_update_fn,
        connectivity_init_fn=connector,
        state_init_fn=make_default_state_init_fn(),
        compute_output_error_fn=make_default_output_error_fn(),
        output_forward_fn=make_default_forward_fn(identity),
        output_neuron_update_fn=make_default_neuron_update_fn(lr, identity),
        output_state_init_fn=make_default_state_init_fn(
            lambda key, shape, dtype, fan_in: jnp.zeros(shape, dtype)
        ),
    )

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

    # Initialize each hidden layer (sequential so each sees prior layers' neurons)
    key = random.PRNGKey(args.seed)
    for _ in range(num_layers):
        key, layer_key = random.split(key)
        network = network.add_layer(n_units=hidden_dim, key=layer_key)

    # Connect last hidden layer to outputs
    key, out_key = random.split(key)
    network = network.connect_to_output(network.get_units_in_layer(-1), out_key)

    return TrainState(network=network, rng=key, step=jnp.array(0))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(train_state: TrainState, num_steps: int) -> TrainState:
    """Train with jax.lax.scan, printing loss every LOG_INTERVAL steps."""
    num_scans = num_steps // LOG_INTERVAL

    @jax.jit
    def scan_steps(state: TrainState):
        def body(state, _):
            key, data_key, step_key = random.split(state.rng, 3)
            x = random.uniform(data_key, (2,), minval=-jnp.pi, maxval=jnp.pi)
            y = jnp.array([jnp.sin(x[0] + x[1]), jnp.cos(x[0] - x[1])])
            network = state.network.step(x, y, step_key)
            loss = jnp.mean((network.output_states.activation_value - y) ** 2)
            return TrainState(network=network, rng=key, step=state.step + 1), loss

        return jax.lax.scan(body, state, length=LOG_INTERVAL)

    for _ in range(num_scans):
        train_state, losses = scan_steps(train_state)
        print(f"Step {int(train_state.step):>6d} | Loss: {float(losses.mean()):.6f}")

    return train_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    train_state = init_experiment(args)

    print(f"Network: {args.num_layers} layer(s), {args.hidden_dim} hidden, lr={args.learning_rate}")
    print(f"Task: y = [sin(x1+x2), cos(x1-x2)]")

    train_state = train(train_state, args.num_steps)

    # Evaluation
    key = train_state.rng
    losses = []
    for _ in range(100):
        key, test_key = random.split(key)
        x = random.uniform(test_key, (2,), minval=-jnp.pi, maxval=jnp.pi)
        y = jnp.array([jnp.sin(x[0] + x[1]), jnp.cos(x[0] - x[1])])
        eval_net = eqx.tree_at(lambda s: s.input_values, train_state.network, x)
        eval_net = eval_net._forward_pass()
        pred = eval_net.output_states.activation_value
        losses.append(float(jnp.mean((pred - y) ** 2)))

    print(f"Test MSE: {sum(losses) / len(losses):.6f}")


if __name__ == "__main__":
    main()
