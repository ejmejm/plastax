"""Demo: Standard MLP with MSE + SGD on synthetic regression using the plastax framework.

Task: y = [sin(x1 + x2), cos(x1 - x2)], 2 inputs, 2 outputs.
Uses default forward/backward/update functions from plastax.defaults.
"""

import argparse
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
import mlflow
import numpy as np
from tqdm import tqdm

from plastax import (
    DefaultNeuronState,
    Network,
    StateUpdateFunctions,
    StructureUpdateState,
    default_structure_update_fn,
    make_default_backward_signal_fn,
    make_default_forward_fn,
    make_default_init_neuron_fn,
    make_default_neuron_update_fn,
    make_default_output_error_fn,
)


UNROLL_STEPS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Dynamic net demo: MLP on sin/cos regression')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=50_000)
    parser.add_argument('--log_interval', type=int, default=500)
    parser.add_argument('--mlflow', action='store_true')
    return parser.parse_args()


# ---------------------------------------------------------------------------
# TrainState and StepMetrics
# ---------------------------------------------------------------------------

class DemoStructureUpdateState(StructureUpdateState):
    """Need at least one field for scan compatibility."""
    dummy: jax.Array


class TrainState(eqx.Module):
    # Static
    log_interval: int = eqx.field(static=True)

    # Dynamic
    network: Network
    rng: jax.Array
    step: jax.Array


class StepMetrics(eqx.Module):
    """Metrics collected from a single training step."""
    loss: jax.Array


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_experiment(args: argparse.Namespace) -> TrainState:
    """Initialize network, user functions, and training state."""
    seed = args.seed if args.seed is not None else np.random.randint(0, 1_000_000_000)
    key = random.PRNGKey(seed)

    n_inputs = 2
    n_outputs = 2
    hidden_dim = args.hidden_dim
    num_layers = args.num_layers
    # Layer 0 connects to inputs; deeper layers connect to prior hidden layer.
    max_hidden_conn = max(n_inputs, hidden_dim)

    # Define neuron classes with baked-in max_connections
    class HiddenNeuron(DefaultNeuronState):
        def __init__(self):
            super().__init__(max_connections=max_hidden_conn)

    class OutputNeuron(DefaultNeuronState):
        def __init__(self):
            super().__init__(max_connections=hidden_dim)

    # Build state update functions
    lr = args.learning_rate
    identity = lambda x: x

    state_update_fns = StateUpdateFunctions(
        forward_fn=make_default_forward_fn(jax.nn.relu),
        backward_signal_fn=make_default_backward_signal_fn(),
        neuron_update_fn=make_default_neuron_update_fn(lr, jax.nn.relu),
        structure_update_fn=default_structure_update_fn,
        init_neuron_fn=make_default_init_neuron_fn(HiddenNeuron),
        compute_output_error_fn=make_default_output_error_fn(),
        output_forward_fn=make_default_forward_fn(identity),
        output_neuron_update_fn=make_default_neuron_update_fn(lr, identity),
    )

    structure_state = DemoStructureUpdateState(dummy=jnp.array(0))

    network = Network(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        max_hidden_per_layer=hidden_dim,
        max_layers=args.num_layers,
        max_generate_per_step=0,
        auto_connect_to_output=False,
        hidden_neuron_cls=HiddenNeuron,
        output_neuron_cls=OutputNeuron,
        state_update_fns=state_update_fns,
        structure_state=structure_state,
    )

    # Activate all hidden neurons
    hidden_active = jnp.ones(network.total_hidden, dtype=bool)
    network = eqx.tree_at(
        lambda s: s.hidden_states.active_mask,
        network,
        hidden_active,
    )

    # Connect each hidden layer to the prior layer (layer 0 -> inputs, layer k -> layer k-1)
    max_conn = network.max_connections
    total_hidden = network.total_hidden
    incoming_ids = jnp.zeros((total_hidden, max_conn), dtype=jnp.int32)
    active_conn_mask = jnp.zeros((total_hidden, max_conn), dtype=bool)

    for k in range(num_layers):
        start = k * hidden_dim
        end = (k + 1) * hidden_dim
        if k == 0:
            ids = jnp.broadcast_to(
                jnp.arange(n_inputs, dtype=jnp.int32),
                (hidden_dim, n_inputs),
            )
            incoming_ids = incoming_ids.at[start:end, :n_inputs].set(ids)
            active_conn_mask = active_conn_mask.at[start:end, :n_inputs].set(True)
        else:
            prev_global_start = n_inputs + (k - 1) * hidden_dim
            ids = jnp.broadcast_to(
                jnp.arange(hidden_dim, dtype=jnp.int32) + prev_global_start,
                (hidden_dim, hidden_dim),
            )
            incoming_ids = incoming_ids.at[start:end, :hidden_dim].set(ids)
            active_conn_mask = active_conn_mask.at[start:end, :hidden_dim].set(True)

    network = eqx.tree_at(
        lambda s: s.hidden_states.incoming_ids,
        network,
        incoming_ids,
    )
    network = eqx.tree_at(
        lambda s: s.hidden_states.active_connection_mask,
        network,
        active_conn_mask,
    )

    # Connect output neurons only to the last hidden layer
    max_out_conn = network.max_output_connections
    last_layer_global_start = n_inputs + (num_layers - 1) * hidden_dim
    output_incoming_ids = jnp.broadcast_to(
        jnp.arange(hidden_dim, dtype=jnp.int32) + last_layer_global_start,
        (n_outputs, hidden_dim),
    )
    if hidden_dim < max_out_conn:
        output_incoming_ids = jnp.pad(
            output_incoming_ids, ((0, 0), (0, max_out_conn - hidden_dim)))
    output_conn_mask = jnp.zeros((n_outputs, max_out_conn), dtype=bool)
    output_conn_mask = output_conn_mask.at[:, :hidden_dim].set(True)

    network = eqx.tree_at(
        lambda s: s.output_states.incoming_ids,
        network,
        output_incoming_ids,
    )
    network = eqx.tree_at(
        lambda s: s.output_states.active_connection_mask,
        network,
        output_conn_mask,
    )

    # Initialize weights with Xavier-like scaling per layer
    hidden_weights = jnp.zeros((total_hidden, max_conn))
    for k in range(num_layers):
        start = k * hidden_dim
        end = (k + 1) * hidden_dim
        if k == 0:
            scale = jnp.sqrt(2.0 / n_inputs)
            n_conn = n_inputs
        else:
            scale = jnp.sqrt(2.0 / hidden_dim)
            n_conn = hidden_dim
        key, w_key = random.split(key)
        w = random.normal(w_key, (hidden_dim, n_conn)) * scale
        hidden_weights = hidden_weights.at[start:end, :n_conn].set(w)

    key, w_output_key = random.split(key)
    output_weight_scale = jnp.sqrt(2.0 / hidden_dim)
    output_weights = jnp.zeros((n_outputs, max_out_conn))
    output_weights = output_weights.at[:, :hidden_dim].set(
        random.normal(w_output_key, (n_outputs, hidden_dim)) * output_weight_scale)

    network = eqx.tree_at(
        lambda s: s.hidden_states.weights,
        network,
        hidden_weights,
    )
    network = eqx.tree_at(
        lambda s: s.output_states.weights,
        network,
        output_weights,
    )

    return TrainState(
        log_interval=args.log_interval,
        network=network,
        rng=key,
        step=jnp.array(0),
    )


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(train_state: TrainState, _) -> Tuple[TrainState, StepMetrics]:
    """Single training step for jax.lax.scan."""
    key, data_key, step_key = random.split(train_state.rng, 3)

    # Sample one data point
    x = random.uniform(data_key, (2,), minval=-jnp.pi, maxval=jnp.pi)
    x1, x2 = x[0], x[1]
    y = jnp.array([jnp.sin(x1 + x2), jnp.cos(x1 - x2)])

    # Run framework step
    network = train_state.network.step(x, y, step_key)

    # Compute loss for logging
    output_activations = network.output_states.activation_value
    loss = jnp.mean((output_activations - y) ** 2)

    new_state = TrainState(
        log_interval=train_state.log_interval,
        network=network,
        rng=key,
        step=train_state.step + 1,
    )

    return new_state, StepMetrics(loss=loss)


# ---------------------------------------------------------------------------
# Logging and training loop
# ---------------------------------------------------------------------------

def log_metrics(metrics: StepMetrics, step: int, use_mlflow: bool = True) -> float:
    """Aggregate scan-batch metrics and log to MLflow."""
    mean_loss = float(metrics.loss.mean())
    if use_mlflow:
        mlflow.log_metrics({'loss': mean_loss}, step=step)
    return mean_loss


def train(train_state: TrainState, num_steps: int, use_mlflow: bool = False) -> TrainState:
    """Outer loop over jax.lax.scan inner loop."""
    log_interval = train_state.log_interval
    num_scans = num_steps // log_interval

    @jax.jit
    def scan_steps(state: TrainState) -> Tuple[TrainState, StepMetrics]:
        return jax.lax.scan(
            train_step,
            state,
            length=log_interval,
            unroll=UNROLL_STEPS,
        )

    pbar = tqdm(total=num_steps, desc='Training')

    for _ in range(num_scans):
        train_state, metrics = scan_steps(train_state)

        mean_loss = log_metrics(
            metrics, step=int(train_state.step.item()), use_mlflow=use_mlflow)
        pbar.update(log_interval)
        pbar.set_postfix({'loss': f'{mean_loss:.6f}'})

    pbar.close()
    return train_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    train_state = init_experiment(args)

    print(f'Network: {args.num_layers} layer(s), {args.hidden_dim} hidden units, lr={args.learning_rate}')
    print(f'Task: y = [sin(x1+x2), cos(x1-x2)]')

    if args.mlflow:
        mlflow.start_run()
        mlflow.log_params(vars(args))

    train_state = train(train_state, args.num_steps, use_mlflow=args.mlflow)

    # Final evaluation
    key = train_state.rng
    test_losses = []
    for _ in range(100):
        key, test_key = random.split(key)
        x = random.uniform(test_key, (2,), minval=-jnp.pi, maxval=jnp.pi)
        y = jnp.array([jnp.sin(x[0] + x[1]), jnp.cos(x[0] - x[1])])
        # Quick forward pass for evaluation
        eval_net = eqx.tree_at(
            lambda s: s.input_values,
            train_state.network,
            x,
        )
        eval_net = eval_net._forward_pass()
        pred = eval_net.output_states.activation_value
        test_losses.append(float(jnp.mean((pred - y) ** 2)))

    avg_test_loss = float(np.mean(test_losses))
    print(f'Average test MSE: {avg_test_loss:.6f}')

    if args.mlflow:
        mlflow.log_metrics({'test_loss': avg_test_loss})
        mlflow.end_run()

    print('Training complete!')


if __name__ == '__main__':
    main()
