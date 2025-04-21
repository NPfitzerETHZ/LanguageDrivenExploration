import equinox as eqx
import jax.numpy as jnp
import jax
from heading_ideas import Heading
from sentence_transformers import SentenceTransformer
from modules import Block
llm = SentenceTransformer('thenlper/gte-large')
device = "mps"

class Decoder(eqx.Module):
    l0: eqx.nn.Linear
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear
    l3: eqx.nn.Linear

    def __init__(self, emb_size):
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        hidden_size = 256
        self.l0 = Block(emb_size, hidden_size, 0, key=keys[0])
        self.l1 = Block(hidden_size, hidden_size, 0, key=keys[1])
        self.l2 = Block(hidden_size, hidden_size, 0, key=keys[2])
        self.l3 = eqx.nn.Linear(hidden_size, 2, key=keys[3])

    def __call__(self, embed):
        x = self.l0(embed)
        x = self.l1(x)
        x = self.l2(x)
        return self.l3(x)


# Load the trained model from disk
model_path = "llm1_decoder_model.eqx"  # replace with your actual path
embedding_size = llm.encode(["test sentence"], device = device).shape[1]
model = Decoder(embedding_size)  # Replace with your hidden size
model = eqx.tree_deserialise_leaves(model_path, model)

# Example inference
def predict(model, embeddings):
    predictions = jax.vmap(model)(embeddings)
    return predictions

# Example inference
test_phrase = ["the target is in the right middle edge."]
test_embedding = llm.encode(test_phrase).squeeze(0)
prediction = model(test_embedding)

print(
    {      
        "test_phrase": test_phrase,
        "result position": prediction
    }
)
