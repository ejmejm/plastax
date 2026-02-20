"""Tests for forward pass, backward pass, and learning correctness.

Includes the MLP equivalence test that compares plastax weight trajectories
against an equivalent Equinox MLP with manual SGD.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from plastax import (
    BackpropNeuronState,
    Network,
    StructureUpdateState,
    tree_replace,
)
from plastax.standard import (
    make_backprop_sgd_update_functions,
    make_prior_layer_connector,
    make_weighted_sum_forward_fn,
)

from conftest import (
    SmallHiddenNeuron,
    SmallOutputNeuron,
    extract_weight_matrix,
    forward_output,
    make_forward_only_fns,
)


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def test_relu_forward_pass(key):
    """ReLU forward pass produces correct output, including the zero case."""
    relu_forward = make_weighted_sum_forward_fn(jax.nn.relu)
    identity_forward = make_weighted_sum_forward_fn(lambda x: x)

    fns = make_forward_only_fns(
        forward_fn=relu_forward,
        output_forward_fn=identity_forward,
    )

    net = Network(
        n_inputs=2,
        n_outputs=1,
        max_hidden_per_layer=4,
        max_layers=1,
        hidden_neuron_cls=SmallHiddenNeuron,
        output_neuron_cls=SmallOutputNeuron,
        state_update_fns=fns,
        structure_state=StructureUpdateState(),
    )

    # Add hidden neuron with specific weights [0.5, -0.5, 0, 0]
    incoming = jnp.array([0, 1, -1, -1], dtype=jnp.int32)
    key, k1, kf1, kf2 = jax.random.split(key, 4)
    net, _ = net.add_unit(incoming, k1, connect_to_output=True)

    # Override hidden weights to [0.5, -0.5]
    net = tree_replace(
        net,
        hidden_states=tree_replace(
            net.hidden_states,
            weights=net.hidden_states.weights.at[0, :2].set(
                jnp.array([0.5, -0.5])),
        ),
    )
    # Override output weight to 2.0
    net = tree_replace(
        net,
        output_states=tree_replace(
            net.output_states,
            weights=net.output_states.weights.at[0].set(
                jnp.where(net.output_states.get_active_connection_mask()[0], 2.0, 0.0)),
        ),
    )

    # Positive pre-activation: 0.5*3 + (-0.5)*1 = 1.0 -> relu = 1.0 -> output = 2.0
    out1 = forward_output(net, jnp.array([3.0, 1.0]), kf1)
    assert jnp.allclose(out1, jnp.array([2.0]), atol=1e-5)

    # Negative pre-activation: 0.5*1 + (-0.5)*3 = -1.0 -> relu = 0.0 -> output = 0.0
    out2 = forward_output(net, jnp.array([1.0, 3.0]), kf2)
    assert jnp.allclose(out2, jnp.array([0.0]), atol=1e-5)


# ---------------------------------------------------------------------------
# Single SGD step
# ---------------------------------------------------------------------------

def test_single_sgd_step():
    """One network.step produces the expected weight change (hand-computed)."""
    n_inputs, n_outputs, hidden_dim = 2, 1, 1
    lr = 0.1

    class H(BackpropNeuronState):
        def __init__(self):
            super().__init__(max_connections=2)

    class O(BackpropNeuronState):
        def __init__(self):
            super().__init__(max_connections=1)

    connector = make_prior_layer_connector(n_inputs, hidden_dim)
    fns = make_backprop_sgd_update_functions(connector, learning_rate=lr)

    net = Network(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        max_hidden_per_layer=hidden_dim,
        max_layers=1,
        hidden_neuron_cls=H,
        output_neuron_cls=O,
        state_update_fns=fns,
        structure_state=StructureUpdateState(),
    )

    key = jax.random.PRNGKey(42)
    key, k1, k2 = jax.random.split(key, 3)
    net = net.add_layer(n_units=1, key=k1)
    net = net.connect_to_output(net.get_units_in_layer(0), k2)

    # Extract initial weights
    w_h_before = extract_weight_matrix(net.hidden_states, n_inputs, hidden_dim)
    w_o_before = extract_weight_matrix(
        net.output_states, n_inputs + hidden_dim, n_outputs)

    # Run one step
    x = jnp.array([1.0, 2.0])
    y = jnp.array([0.0])
    key, sk = jax.random.split(key)
    net = net.step(x, y, sk)

    w_h_after = extract_weight_matrix(net.hidden_states, n_inputs, hidden_dim)
    w_o_after = extract_weight_matrix(
        net.output_states, n_inputs + hidden_dim, n_outputs)

    # Hand-compute expected values
    # Forward: h_pre = w_h @ x, h = relu(h_pre), o = w_o_col @ h
    h_pre = (w_h_before @ x).item()
    h = max(h_pre, 0.0)
    w_o_col = w_o_before[0, n_inputs]  # output weight from hidden neuron
    o = w_o_col * h

    # Output error: 2*(o - y) / n_outputs
    out_err = 2.0 * (o - y[0]) / n_outputs

    # Output delta (linear activation => deriv = 1): delta_o = out_err * 1
    delta_o = out_err

    # Output weight update: w_o -= lr * delta_o * h
    expected_w_o = w_o_col - lr * delta_o * h

    # Hidden error signal: sum of (output_weight * delta_o * active)
    error_h = w_o_col * delta_o

    # Hidden delta: error * relu'(h_pre)
    relu_deriv = 1.0 if h_pre > 0 else 0.0
    delta_h = error_h * relu_deriv

    # Hidden weight update: w_h -= lr * delta_h * x
    expected_w_h = w_h_before[0] - lr * delta_h * x

    assert jnp.allclose(w_h_after[0], expected_w_h, atol=1e-5), (
        f"Hidden weights: got {w_h_after[0]}, expected {expected_w_h}")
    assert jnp.allclose(
        w_o_after[0, n_inputs], expected_w_o, atol=1e-5), (
        f"Output weight: got {w_o_after[0, n_inputs]}, expected {expected_w_o}")


# ---------------------------------------------------------------------------
# MLP equivalence with Equinox
# ---------------------------------------------------------------------------

class TwoLayerMLP(eqx.Module):
    """No-bias 2-hidden-layer MLP for comparison with plastax."""
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear
    linear3: eqx.nn.Linear

    def __call__(self, x):
        h1 = jax.nn.relu(self.linear1(x))
        h2 = jax.nn.relu(self.linear2(h1))
        return self.linear3(h2)


def _mse_loss(model, x, y):
    pred = model(x)
    return jnp.mean((pred - y) ** 2)


def _extract_layer_weights(hidden_states, hidden_dim, layer_idx, n_sources):
    """Extract dense weight matrix for a specific hidden layer.

    Slices hidden_states to the layer, then scatters sparse weights into a
    dense matrix.
    """
    layer_states = jax.tree.map(
        lambda x: x[layer_idx * hidden_dim:(layer_idx + 1) * hidden_dim],
        hidden_states,
    )
    return extract_weight_matrix(layer_states, n_sources, hidden_dim)


def test_mlp_equivalence_with_equinox():
    """Plastax with standard backprop/SGD produces identical weight
    trajectories to an equivalent 2-layer Equinox MLP with manual SGD."""
    n_inputs, n_outputs, hidden_dim = 2, 2, 10
    num_layers = 2
    lr = 0.01
    n_steps = 5
    max_conn = max(n_inputs, hidden_dim)

    class HiddenNeuron(BackpropNeuronState):
        def __init__(self):
            super().__init__(max_connections=max_conn)

    class OutputNeuron(BackpropNeuronState):
        def __init__(self):
            super().__init__(max_connections=hidden_dim)

    # Build plastax network (same as demo.py pattern)
    connector = make_prior_layer_connector(n_inputs, hidden_dim)
    fns = make_backprop_sgd_update_functions(connector, learning_rate=lr)

    key = jax.random.PRNGKey(123)
    key, k_out = jax.random.split(key)

    plastax_net = Network(
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        max_hidden_per_layer=hidden_dim,
        max_layers=num_layers,
        hidden_neuron_cls=HiddenNeuron,
        output_neuron_cls=OutputNeuron,
        state_update_fns=fns,
        structure_state=StructureUpdateState(),
    )
    for _ in range(num_layers):
        key, layer_key = jax.random.split(key)
        plastax_net = plastax_net.add_layer(n_units=hidden_dim, key=layer_key)
    plastax_net = plastax_net.connect_to_output(
        plastax_net.get_units_in_layer(-1), k_out)

    # Index space:
    #   Inputs:  [0, 1]
    #   Layer 0: abs [2 .. 11]
    #   Layer 1: abs [12 .. 21]
    #   Outputs: abs [22, 23]
    layer0_abs_start = n_inputs
    layer1_abs_start = n_inputs + hidden_dim

    # Extract weight matrices from plastax
    # W1: layer 0 neurons connect to inputs (abs 0..1) → (hidden_dim, n_inputs)
    w1_init = _extract_layer_weights(
        plastax_net.hidden_states, hidden_dim, 0, n_inputs)

    # W2: layer 1 neurons connect to layer 0 (abs 2..11) → extract full then slice
    w2_full = _extract_layer_weights(
        plastax_net.hidden_states, hidden_dim, 1, layer1_abs_start)
    w2_init = w2_full[:, layer0_abs_start:layer1_abs_start]

    # W3: output neurons connect to layer 1 (abs 12..21) → extract full then slice
    total_hidden = num_layers * hidden_dim
    w3_full = extract_weight_matrix(
        plastax_net.output_states, n_inputs + total_hidden, n_outputs)
    w3_init = w3_full[:, layer1_abs_start:layer1_abs_start + hidden_dim]

    # Build equivalent Equinox MLP with same initial weights
    dummy_key = jax.random.PRNGKey(0)
    eqx_model = TwoLayerMLP(
        linear1=eqx.nn.Linear(n_inputs, hidden_dim, use_bias=False, key=dummy_key),
        linear2=eqx.nn.Linear(hidden_dim, hidden_dim, use_bias=False, key=dummy_key),
        linear3=eqx.nn.Linear(hidden_dim, n_outputs, use_bias=False, key=dummy_key),
    )
    eqx_model = eqx.tree_at(lambda m: m.linear1.weight, eqx_model, w1_init)
    eqx_model = eqx.tree_at(lambda m: m.linear2.weight, eqx_model, w2_init)
    eqx_model = eqx.tree_at(lambda m: m.linear3.weight, eqx_model, w3_init)

    # Verify initial forward passes match
    test_x = jnp.array([1.0, -0.5])
    key, kf = jax.random.split(key)
    plastax_fwd = forward_output(plastax_net, test_x, kf)
    eqx_fwd = eqx_model(test_x)
    assert jnp.allclose(plastax_fwd, eqx_fwd, atol=1e-5), (
        f"Initial forward mismatch: plastax={plastax_fwd}, eqx={eqx_fwd}")

    # Train both for n_steps on deterministic data
    for step_i in range(n_steps):
        key, data_key, step_key = jax.random.split(key, 3)
        x = jax.random.uniform(data_key, (2,), minval=-1.0, maxval=1.0)
        y = jnp.array([jnp.sin(x[0] + x[1]), jnp.cos(x[0] - x[1])])

        # Plastax step
        plastax_net = plastax_net.step(x, y, step_key)

        # Equinox manual SGD step
        grads = jax.grad(_mse_loss)(eqx_model, x, y)
        eqx_model = jax.tree.map(lambda p, g: p - lr * g, eqx_model, grads)

        # Extract updated plastax weights
        w1_p = _extract_layer_weights(
            plastax_net.hidden_states, hidden_dim, 0, n_inputs)
        w2_p_full = _extract_layer_weights(
            plastax_net.hidden_states, hidden_dim, 1, layer1_abs_start)
        w2_p = w2_p_full[:, layer0_abs_start:layer1_abs_start]
        w3_p_full = extract_weight_matrix(
            plastax_net.output_states, n_inputs + total_hidden, n_outputs)
        w3_p = w3_p_full[:, layer1_abs_start:layer1_abs_start + hidden_dim]

        assert jnp.allclose(w1_p, eqx_model.linear1.weight, atol=1e-5), (
            f"Step {step_i}: layer 0 weight mismatch.\n"
            f"  max diff: {jnp.max(jnp.abs(w1_p - eqx_model.linear1.weight))}")
        assert jnp.allclose(w2_p, eqx_model.linear2.weight, atol=1e-5), (
            f"Step {step_i}: layer 1 weight mismatch.\n"
            f"  max diff: {jnp.max(jnp.abs(w2_p - eqx_model.linear2.weight))}")
        assert jnp.allclose(w3_p, eqx_model.linear3.weight, atol=1e-5), (
            f"Step {step_i}: output weight mismatch.\n"
            f"  max diff: {jnp.max(jnp.abs(w3_p - eqx_model.linear3.weight))}")
