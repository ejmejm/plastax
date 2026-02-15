"""Demo: Standard MLP with MSE + SGD on synthetic regression using the dynamic_net framework.

Task: y = [sin(x1 + x2), cos(x1 - x2)], 2 inputs, 2 outputs.
Uses default forward/backward/update functions from dynamic_net.defaults.
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

from dynamic_net import (
    Network,
    StructureUpdateState,
    UserFunctions,
    default_structure_update_fn,
    make_default_backward_signal_fn,
    make_default_forward_fn,
    make_default_init_neuron_fn,
    make_default_neuron_state,
    make_default_neuron_update_fn,
    make_default_output_error_fn,
    make_default_output_neuron_state,
)


UNROLL_STEPS = 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Dynamic net demo: MLP on sin/cos regression')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=32)
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
    user_fns: UserFunctions = eqx.field(static=True)

    # Dynamic
    network: Network
    structure_state: DemoStructureUpdateState
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
    max_connections = n_inputs  # hidden units connect to all inputs
    max_output_connections = hidden_dim  # output connects to all hidden units

    # Create neuron templates using defaults (with DefaultForwardPassState)
    hidden_template = make_default_neuron_state(max_connections)
    output_template = make_default_output_neuron_state(max_output_connections)

    key, net_key = random.split(key)
    network = Network.create(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        max_hidden_per_layer=hidden_dim,
        max_layers=1,
        max_connections=max_connections,
        max_output_connections=max_output_connections,
        max_generate_per_step=0,
        auto_connect_to_output=False,
        hidden_neuron_template=hidden_template,
        output_neuron_template=output_template,
        key=net_key,
    )

    # Activate all hidden neurons and connect them to inputs
    hidden_active = jnp.ones(network.total_hidden, dtype=bool)
    network = eqx.tree_at(
        lambda s: s.hidden_states.active_mask,
        network,
        hidden_active,
    )

    # Set hidden neurons' incoming connections to the 2 input neurons
    input_ids = jnp.broadcast_to(
        jnp.arange(n_inputs, dtype=jnp.int32),
        (network.total_hidden, max_connections),
    )
    input_conn_mask = jnp.ones((network.total_hidden, max_connections), dtype=bool)
    network = eqx.tree_at(
        lambda s: s.hidden_states.connectivity.incoming_ids,
        network,
        input_ids,
    )
    network = eqx.tree_at(
        lambda s: s.hidden_states.connectivity.active_connection_mask,
        network,
        input_conn_mask,
    )

    # Connect output neurons to all hidden neurons
    hidden_abs_start = network.n_inputs
    output_incoming_ids = jnp.broadcast_to(
        jnp.arange(hidden_dim, dtype=jnp.int32) + hidden_abs_start,
        (n_outputs, hidden_dim),
    )
    # Pad to max_output_connections if needed
    if hidden_dim < max_output_connections:
        pad_width = max_output_connections - hidden_dim
        output_incoming_ids = jnp.pad(output_incoming_ids, ((0, 0), (0, pad_width)))
    output_conn_mask = jnp.zeros((n_outputs, max_output_connections), dtype=bool)
    output_conn_mask = output_conn_mask.at[:, :hidden_dim].set(True)

    network = eqx.tree_at(
        lambda s: s.output_states.connectivity.incoming_ids,
        network,
        output_incoming_ids,
    )
    network = eqx.tree_at(
        lambda s: s.output_states.connectivity.active_connection_mask,
        network,
        output_conn_mask,
    )

    # Initialize weights with small random values (Xavier-like)
    key, w_hidden_key, w_output_key = random.split(key, 3)
    hidden_weight_scale = jnp.sqrt(2.0 / n_inputs)
    output_weight_scale = jnp.sqrt(2.0 / hidden_dim)

    hidden_weights = random.normal(w_hidden_key, (network.total_hidden, max_connections)) * hidden_weight_scale
    output_weights = jnp.zeros((n_outputs, max_output_connections))
    output_weights = output_weights.at[:, :hidden_dim].set(
        random.normal(w_output_key, (n_outputs, hidden_dim)) * output_weight_scale)

    network = eqx.tree_at(
        lambda s: s.hidden_states.connectivity.weights,
        network,
        hidden_weights,
    )
    network = eqx.tree_at(
        lambda s: s.output_states.connectivity.weights,
        network,
        output_weights,
    )

    # Build user functions
    lr = args.learning_rate
    identity = lambda x: x

    user_fns = UserFunctions(
        forward_fn=make_default_forward_fn(jax.nn.relu),
        backward_signal_fn=make_default_backward_signal_fn(),
        neuron_update_fn=make_default_neuron_update_fn(lr, jax.nn.relu),
        structure_update_fn=default_structure_update_fn,
        init_neuron_fn=make_default_init_neuron_fn(max_connections),
        compute_output_error_fn=make_default_output_error_fn(),
        output_forward_fn=make_default_forward_fn(identity),
        output_neuron_update_fn=make_default_neuron_update_fn(lr, identity),
    )

    structure_state = DemoStructureUpdateState(dummy=jnp.array(0))

    return TrainState(
        log_interval=args.log_interval,
        user_fns=user_fns,
        network=network,
        structure_state=structure_state,
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
    network, structure_state = train_state.network.step(
        x, y, train_state.structure_state, train_state.user_fns, step_key,
    )

    # Compute loss for logging
    output_activations = network.output_states.forward_state.activation_value
    loss = jnp.mean((output_activations - y) ** 2)

    new_state = TrainState(
        log_interval=train_state.log_interval,
        user_fns=train_state.user_fns,
        network=network,
        structure_state=structure_state,
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

    print(f'Network: {args.hidden_dim} hidden units, lr={args.learning_rate}')
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
            lambda s: s.input_activations,
            train_state.network,
            x,
        )
        eval_net = eval_net._forward_pass(train_state.user_fns.forward_fn,
                                          train_state.user_fns.output_forward_fn)
        pred = eval_net.output_states.forward_state.activation_value
        test_losses.append(float(jnp.mean((pred - y) ** 2)))

    avg_test_loss = float(np.mean(test_losses))
    print(f'Average test MSE: {avg_test_loss:.6f}')

    if args.mlflow:
        mlflow.log_metrics({'test_loss': avg_test_loss})
        mlflow.end_run()

    print('Training complete!')


if __name__ == '__main__':
    main()
