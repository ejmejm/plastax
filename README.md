# plastax — Dynamic Neural Network Framework

A JAX/Equinox framework for neural networks that change structure at every step. Each neuron has individual state, and the user defines transition rules for how the network computes, learns, and evolves. The framework simulates these rules.

## Installation

```bash
cd dynamic_neural_networks
pip install -e .

# With demo dependencies:
pip install -e ".[demo]"
```

## Overview

Standard ML frameworks treat the network as a fixed computational graph. This framework treats the network as a **finite state machine** where:

- Each neuron has a `NeuronState` containing connectivity, forward pass data, and backward pass data
- The user defines **transition rules** (functions) that determine how neurons compute, how signals propagate backward, how weights update, and how structure changes
- The framework provides the **simulation loop** that applies these rules layer by layer

The framework provides default implementations of the user functions that implement standard backpropagation with SGD, so it can be used as a standard MLP out of the box while allowing full customization.

### Execution Model

The framework uses **global mode**: forward and backward passes are computed layer by layer from input to output and output to input respectively. A pipelined mode (all neurons simultaneously, using prior-timestep values) may be added in the future.

### Neuron Array Layout

All neurons live in a flat index space:
```
[0 : n_inputs]                     = input neurons (activation values only)
[n_inputs : n_inputs+total_hidden] = hidden neurons (full NeuronState)
[n_inputs+total_hidden : total]    = output neurons (full NeuronState, separate storage)
```

Hidden and output neurons are stored separately because output neurons may need more incoming connections (up to all neurons in the network).

## Core Concepts

### States

**`ConnectivityState`** — Per-neuron incoming connection info:
- `incoming_ids`: indices of source neurons in the flat array
- `weights`: connection weights
- `active_connection_mask`: which connections are active

**`ForwardPassState`** — Per-neuron forward pass data:
- `activation_value`: the neuron's output (required)
- Subclass to add fields like `pre_activation`, `incoming_activations`, etc.

**`BackwardPassState`** — Per-neuron backward pass data:
- `error_signal`: gradient/error signal for this neuron
- Subclass to add custom fields

**`NeuronState`** — Complete per-neuron state:
- `active_mask`: whether this neuron is active
- `connectivity`: ConnectivityState
- `forward_state`: ForwardPassState
- `backward_state`: BackwardPassState
- Subclass to add metadata (e.g., unit utility)

**`StructureUpdateState`** — Metadata for structure decisions. Empty base class; define all fields yourself.

**`NetworkState`** — The full network:
- `input_activations`: input values
- `hidden_states`: stacked NeuronStates for all hidden neurons
- `output_states`: stacked NeuronStates for all output neurons
- `config`: NetworkConfig

### Step Loop

Each step of the framework runs:
1. **Forward pass** — layer by layer, gather incoming activations, call user's forward function
2. **Backward signal** — compute output error, then layer by layer (output → first hidden), propagate error signals via user's backward signal function
3. **Neuron update** — per layer, apply user's update function (e.g., weight updates)
4. **Structure update** — per hidden layer (last to first), user decides which neurons to prune and how many to generate
5. **Neuron generation** — insert new neurons into inactive slots

## User-Defined Functions

All functions are passed via `UserFunctions`. Default implementations are provided in `plastax.defaults`.

### `forward_fn(neuron_state, incoming_activations) -> (activation_value, updated_neuron_state)`

Called per neuron (vmapped over the layer). The framework gathers `incoming_activations` from the flat activation array using the neuron's `incoming_ids`. The user decides how to aggregate them (e.g., weighted sum) and what activation to apply.

### `backward_signal_fn(neuron_state, neuron_index, next_layer_states) -> BackwardPassState`

Called per neuron (vmapped). Receives this neuron's state, its absolute index, and the full NeuronStates of the next layer (already processed). The user searches `next_layer_states.connectivity.incoming_ids` for `neuron_index` to find outgoing connections and compute the error signal arriving at this neuron.

### `neuron_update_fn(neuron_state) -> NeuronState`

Called per neuron (vmapped) after `backward_signal_fn`. The backward state has been set with the propagated error. The user applies activation derivatives, updates weights, and stores the final delta.

### `compute_output_error_fn(output_activations, targets) -> output_errors`

Computes the error vector for the output layer (e.g., MSE derivative).

### `structure_update_fn(layer_states, next_layer_states, structure_state) -> (structure_state, prune_mask, n_generate)`

Decides which neurons to prune and how many to generate per layer.

### `init_neuron_fn(connectivity_state, index, key) -> NeuronState`

Initializes a newly generated neuron given its connectivity.

### Output-Specific Overrides

`UserFunctions` has optional `output_forward_fn` and `output_neuron_update_fn` fields. When set, the output layer uses these instead of the general functions. This is useful when the output layer has a different activation (e.g., linear instead of ReLU).

## User-Defined States

Extend the base states by subclassing:

```python
from plastax import ForwardPassState

class MyForwardPassState(ForwardPassState):
    pre_activation: Float[Array, '']
    incoming_activations: Float[Array, 'max_connections']
```

All neurons must share the same state structure (required for vmapping). Define the class once and all neurons use it.

## Default Implementations

`plastax.defaults` provides factory functions for standard backprop + SGD:

- **`make_default_forward_fn(activation_fn)`** — Weighted sum + activation. Stores `pre_activation` and `incoming_activations` in `DefaultForwardPassState`.
- **`make_default_backward_signal_fn()`** — Standard backprop: finds outgoing connections in the next layer and sums `weight * error_signal * mask`.
- **`make_default_neuron_update_fn(learning_rate, activation_fn)`** — Applies activation derivative, updates weights via SGD, stores delta in `error_signal`.
- **`make_default_output_error_fn()`** — MSE derivative: `2 * (pred - target) / n_outputs`.
- **`default_structure_update_fn`** — No-op (no pruning or generation).

## SGD Walkthrough

Here's how a single step works with the default functions on a 2-layer network (input → hidden → output):

### Forward Pass

1. **Framework** sets `input_activations = x`
2. **Framework** gathers incoming activations for each hidden neuron: `all_activations[incoming_ids]`
3. **User's forward_fn** (hidden layer):
   - `pre_activation = (incoming_acts * weights * mask).sum()`
   - `activation_value = relu(pre_activation)`
   - Stores `pre_activation`, `incoming_activations`, `activation_value` in forward state
4. **Framework** updates activation array with hidden outputs, gathers for output layer
5. **User's forward_fn** (output layer, with identity activation):
   - `pre_activation = (incoming_acts * weights * mask).sum()`
   - `activation_value = pre_activation` (linear)

### Backward Pass

6. **User's compute_output_error_fn**: `output_error = 2 * (pred - target) / n_outputs`
7. **Framework** sets `output_states.backward_state.error_signal = output_error`
8. **User's neuron_update_fn** (output layer):
   - `act_deriv = 1.0` (identity derivative)
   - `delta = error_signal * act_deriv`
   - `weights -= lr * delta * incoming_activations`
   - Stores `delta` in `error_signal`
9. **User's backward_signal_fn** (hidden layer):
   - Searches output layer's `incoming_ids` for this neuron's index
   - `error_from_above = sum(matching_weight * output_delta * mask)`
   - Stores in `error_signal`
10. **User's neuron_update_fn** (hidden layer):
    - `act_deriv = (pre_activation > 0).astype(float)` (ReLU derivative)
    - `delta = error_signal * act_deriv`
    - `weights -= lr * delta * incoming_activations`
    - Stores `delta` in `error_signal`

### Structure Update

11. **Default**: no-op (no pruning or generation)

## Framework Operations

Pure functions for modifying network structure:

- `add_unit(network_state, layer_idx, neuron_state)` — Insert a neuron into an inactive slot
- `remove_unit(network_state, neuron_abs_idx)` — Deactivate a neuron and clean up connections
- `add_connection_to_hidden(network_state, from_idx, to_hidden_rel, weight)` — Add a connection to a hidden neuron
- `add_connection_to_output(network_state, from_idx, to_output_rel, weight)` — Add a connection to an output neuron
- `remove_connection_from_hidden(network_state, hidden_rel, slot)` — Remove a hidden neuron's connection
- `remove_connection_from_output(network_state, output_rel, slot)` — Remove an output neuron's connection

## Running the Demo

```bash
# Install with demo dependencies
pip install -e ".[demo]"

# Run the demo (MLP on sin/cos regression)
python demo.py --seed 42 --num_steps 50000 --hidden_dim 32 --learning_rate 0.01

# With MLflow logging
python demo.py --seed 42 --num_steps 50000 --mlflow
```
