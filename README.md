# plastax — Dynamic Neural Network Framework

A JAX/Equinox framework for neural networks that change structure at every step. Each neuron has individual state, and the user defines transition rules for how the network computes, learns, and evolves. The framework simulates these rules.

## Installation

```bash
pip install -e .

# With demo dependencies:
pip install -e ".[demo]"
```

## Overview

Standard ML frameworks treat the network as a fixed computational graph. This framework treats the network as a **finite state machine** where:

- Each neuron has a `NeuronState` containing connectivity, activation, and error data
- The user defines **transition rules** (functions) that determine how neurons compute, how signals propagate backward, how weights update, and how structure changes
- The framework provides the **simulation loop** that applies these rules layer by layer

The framework provides default implementations that implement standard backpropagation with SGD, so it can be used as a standard MLP out of the box while allowing full customization.

### Execution Model

The framework uses **global mode**: forward and backward passes are computed layer by layer from input to output and output to input respectively. A pipelined mode (all neurons simultaneously, using prior-timestep values) may be added in the future.

### Neuron Array Layout

All neurons live in a flat index space:
```
[0 : n_inputs]                     = input neurons (values only, no NeuronState)
[n_inputs : n_inputs+total_hidden] = hidden neurons (full NeuronState)
[n_inputs+total_hidden : total]    = output neurons (full NeuronState, separate storage)
```

Hidden and output neurons are stored separately because output neurons may use a different neuron class (e.g., different max_connections).

## Core Concepts

### States

**`NeuronState`** — Complete per-neuron state. Users subclass to add extra fields:
- `active_mask`: whether this neuron is active
- `incoming_ids`: indices of source neurons in the flat index space
- `weights`: connection weights
- `active_connection_mask`: which incoming connections are active
- `activation_value`: the neuron's output
- `error_signal`: gradient/error signal

All fields except `active_mask` have a `max_connections` dimension set in the constructor. Neuron classes should have **no required constructor arguments** — `max_connections` is baked in as a default:

```python
class HiddenNeuron(DefaultNeuronState):
    def __init__(self):
        super().__init__(max_connections=16)
```

**`StructureUpdateState`** — Metadata for structure decisions. Empty base class; define all fields yourself.

### Network

`Network` is an Equinox module that holds the full state. It takes neuron **classes** (not instances) and vmaps over their constructors to build initial state arrays. `max_connections` and `max_output_connections` are derived from the constructed neurons' shapes.

```python
network = Network(
    n_inputs=4,
    n_outputs=2,
    max_hidden_per_layer=32,
    max_layers=3,
    hidden_neuron_cls=HiddenNeuron,
    output_neuron_cls=OutputNeuron,   # defaults to hidden_neuron_cls
    state_update_fns=fns,
    structure_state=StructureUpdateState(),
    max_generate_per_step=4,
    auto_connect_to_output=True,      # wire output neurons to inputs on init
)
```

### Step Loop

Each call to `network.step(inputs, targets, key)` runs:

1. **Forward pass** — layer by layer (input to output), gather incoming activations, call `forward_fn`
2. **Backward pass** — compute output error, then top-down: for each layer, propagate backward signals using the layer above's original weights, then update that layer's weights. This interleaves signal propagation one step ahead of weight updates so each layer's signals are computed before its weights change.
3. **Structure update** — per hidden layer (last to first): decide which neurons to prune and how many to generate, apply pruning, then immediately generate new neurons into that layer. Generation happens inline so lower layers can see updated state from upper layers.

## User-Defined Functions

All functions are passed via `StateUpdateFunctions`. Default implementations are provided in `plastax.defaults`.

### `forward_fn(neuron_state, incoming_activations) -> (activation_value, updated_neuron_state)`

Called per neuron (vmapped over the layer). The framework gathers `incoming_activations` from the flat activation array using the neuron's `incoming_ids`. The user decides how to aggregate them (e.g., weighted sum) and what activation to apply.

### `backward_signal_fn(neuron_state, neuron_index, next_layer_states) -> error_signal`

Called per neuron (vmapped). Receives this neuron's state, its absolute index, and the full NeuronStates of the next layer (already processed). The user searches `next_layer_states.incoming_ids` for `neuron_index` to find outgoing connections and compute the error signal arriving at this neuron.

### `neuron_update_fn(neuron_state) -> NeuronState`

Called per neuron (vmapped) after backward signals have been propagated. The backward state has been set with the propagated error. The user applies activation derivatives, updates weights, and stores the final delta.

### `compute_output_error_fn(output_activations, targets) -> output_errors`

Computes the error vector for the output layer (e.g., MSE derivative).

### `structure_update_fn(layer_states, next_layer_states, structure_state) -> (structure_state, prune_mask, n_generate)`

Decides which neurons to prune and how many to generate per layer.

### `init_neuron_fn(hidden_states, connectable_mask, index, key) -> NeuronState`

Initializes a newly generated neuron. Receives the full hidden states and a boolean mask over the global index space (`n_inputs + total_hidden`) indicating which neurons the new neuron is allowed to connect to (all inputs + active hidden neurons in all prior layers). The function decides which neurons to connect to, what weights to use, and how to set up the rest of the neuron state.

The default implementation randomly selects up to `max_connections` neurons from the connectable set.

### Output-Specific Overrides

`StateUpdateFunctions` has optional `output_forward_fn` and `output_neuron_update_fn` fields. When set, the output layer uses these instead of the general functions. This is useful when the output layer has a different activation (e.g., linear instead of ReLU).

## User-Defined States

Extend the base `NeuronState` by subclassing:

```python
class MyNeuronState(NeuronState):
    pre_activation: Float[Array, '']
    incoming_activations: Float[Array, 'max_connections']

    def __init__(self):
        super().__init__(max_connections=16)
        self.pre_activation = jnp.array(0.0)
        self.incoming_activations = jnp.zeros(16)
```

All neurons must share the same state structure (required for vmapping). Define the class once and all neurons use it.

## Default Implementations

`plastax.defaults` provides factory functions for standard backprop + SGD:

- **`DefaultNeuronState`** — `NeuronState` subclass with `pre_activation` and `incoming_activations` fields needed by the default forward/update functions.
- **`make_default_forward_fn(activation_fn)`** — Weighted sum + activation. Stores `pre_activation` and `incoming_activations`.
- **`make_default_backward_signal_fn()`** — Standard backprop: finds outgoing connections in the next layer and sums `weight * error_signal * mask`.
- **`make_default_neuron_update_fn(learning_rate, activation_fn)`** — Applies activation derivative, updates weights via SGD, stores delta in `error_signal`.
- **`make_default_output_error_fn()`** — MSE derivative: `2 * (pred - target) / n_outputs`.
- **`default_structure_update_fn`** — No-op (no pruning or generation).
- **`make_default_init_neuron_fn(neuron_cls)`** — Randomly connects to a subset of connectable neurons with zero weights.

## Network Operations

Methods on `Network` for manually modifying structure:

- `add_unit(neuron_state, connect_to_output)` — Insert a neuron into the correct hidden layer (determined by its incoming connections). Returns `(network, success)`.
- `remove_unit(neuron_abs_idx)` — Deactivate a neuron and clean up dangling connections.
- `add_connection_to_hidden(from_idx, to_idx, weight)` — Add a connection to a hidden neuron. Returns `(network, success)`.
- `add_connection_to_output(from_idx, to_idx, weight)` — Add a connection to an output neuron. Returns `(network, success)`.
- `remove_connection_from_hidden(hidden_rel, slot)` — Remove a hidden neuron's connection.
- `remove_connection_from_output(output_rel, slot)` — Remove an output neuron's connection.

## Running the Demo

```bash
pip install -e .
python demo.py --seed 42 --num_steps 50000 --hidden_dim 32 --learning_rate 0.01
```

### TODO

**Core functional changes:**
- [ ] Add tests for core functionality.
- [ ] Add the ability to set outgoing connections on new unit init.
- [ ] Change backward error signal propagation to allow for propagating more than the error signal.
- [ ] Add pipelined version of the model, which entails making a parent class with everything except for the framework-defined transition functions, then having different children classes that implement those differently for global and pipelined computation.
- [ ] Add a notion of location to units, be it a layer index or an n-dimensional index to pass to connectivity init functions.

**Usability changes:**
- [ ] Change the wording of defaults to clearly indicate that it is doing backprop and standard ML stuff. The word default says nothing about what they do. You can have alternative names that call them default.
- [ ] Have typing for each of the user defined functions, and easy place to go and check what the I/O of each of those functions should be, and something that will allow for type checking.
- [ ] Add examples of different algorithms
   - [ ] SGD (done but need to simplify)
   - [ ] Autostep output layer + SGD elsewhere
   - [ ] Continual backprop
- [x] Make an easy way of initializing individual units or entire layers. These should be defined mainly for individual units, but there should be a very easy way to apply to layers that uses vmapping in the backend. It should be easy to infer how they work on the backend, no magic:
   - Connectivity mode:
      - All prior units (in all prior layers and inputs, starting from latest layer and going backwards until max input connections hit)
      - Random all prior units (randomly in all prior layers and input up to max input connections)
      - All prior layer units (same as first but only units in prior layer)
      - Random prior layer units (same as second but only units in prior layer)
   - Weight initialization:
      - Zeros (maybe this is already the default)
      - Xavier
      - Lecun normal/uniform
      - Kaiming normal/uniform

**Efficiency changes:**
- [ ] Have an option of running the structural change step only once every `n` steps. For this to be jittable without a jax.lax.cond, there will need to be a new step function that actually scans over n steps without the structural change, then does the structural change at the end. The user won't be able to call this function for just 1 step, but that is the price for being able to do this without conditionals. The original step function should still remain, this would just be a second preferred option.
- [ ] Continue allowing users to make individual structural changes, but put them in a buffer until `step` or a new `apply_structure_changes` function is called so that they can be parallelized.
