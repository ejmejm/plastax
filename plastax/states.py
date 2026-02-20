import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int


CONNECTION_PADDING = -1


class NeuronState(eqx.Module):
    """Core state for each neuron. Users subclass to add extra metadata."""
    active_mask: Bool[Array, '']
    incoming_ids: Int[Array, 'max_connections']
    weights: Float[Array, 'max_connections']
    activation_value: Float[Array, '']
    error_signal: Float[Array, '']

    def __init__(self, max_connections: int):
        self.active_mask = jnp.array(False)
        self.incoming_ids = jnp.full(max_connections, CONNECTION_PADDING, dtype=jnp.int32)
        self.weights = jnp.zeros(max_connections)
        self.activation_value = jnp.array(0.0)
        self.error_signal = jnp.array(0.0)
        
    def get_active_connection_mask(self) -> Bool[Array, 'max_connections']:
        return self.incoming_ids >= 0


class StructureUpdateState(eqx.Module):
    """State for structure update decisions. Users define all fields."""
    pass


def tree_replace(tree: eqx.Module, **kwargs) -> eqx.Module:
    """Replace fields of an Equinox module by name."""
    values = [kwargs[k] for k in kwargs]
    return eqx.tree_at(lambda x: [getattr(x, k) for k in kwargs], tree, values)
