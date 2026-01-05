import streamlit as st
import matplotlib.pyplot as plt
from attention import Attention, SampleTokenizer, set_random_seed
import pandas as pd

st.set_page_config(page_title="Attention Mechanism Demo", layout="wide")
st.title("Attention Mechanism Demo")

st.caption("This demo showcases a simple attention mechanism implemented in PyTorch.")

# Input section for user to enter a sentence
user_sentence = st.text_area(
    "Enter a sentence to process with the attention mechanism:",
    "this is a sample sentence for attention mechanism",
    height=100,
)

hidden_dim = st.slider(
    "Select hidden dimension size:", min_value=5, max_value=50, value=10, step=5
)

# selection for random seed
random_seed = st.number_input(
    "Set random seed for reproducibility:",
    min_value=0,
    max_value=1000,
    value=42,
    step=1,
)

# button to trigger sentence processing
if st.button("Process Sentence"):
    # Validate user input
    if not user_sentence.strip():
        st.error("Please enter a sentence before processing.")
    elif len(user_sentence.split()) > 50:
        st.error("Sentence is too long. Please limit your input to 50 words or less.")
    else:
        set_random_seed(random_seed)
        tokenizer = SampleTokenizer(user_sentence, hidden_dim)
        attention_layer = Attention(dim_in=hidden_dim, dim_out=hidden_dim)
        output = tokenizer.embedded.unsqueeze(0)
        attention_score = attention_layer.forward(output)

        st.subheader("Results")

        # showcasing final vocab in table form with keys and values
        st.write("Vocabulary:")
        vocab_items = [
            {"Word": word, "Index": idx} for word, idx in tokenizer.vocab.items()
        ]
        st.table(vocab_items)

        # displaying the attention scores as a heatmap
        st.write("Self-Attention Scores Heatmap:")
        fig, ax = plt.subplots()
        fig.patch.set_alpha(0)  # Transparent figure background
        ax.patch.set_alpha(0)  # Transparent axes background
        cax = ax.matshow(attention_score[0].detach().numpy(), cmap="YlOrRd")
        cbar = fig.colorbar(cax)
        cbar.ax.yaxis.set_tick_params(color="white")
        # cbar.outline.set_edgecolor("white")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")
        ax.set_xticks(range(len(tokenizer.encoded)))
        ax.set_yticks(range(len(tokenizer.encoded)))
        ax.set_xticklabels(user_sentence.split(" "), rotation=90, color="white")
        ax.set_yticklabels(user_sentence.split(" "), color="white")
        ax.tick_params(axis="both", colors="white")  # Change tick marks color to white
        st.pyplot(fig, transparent=True)
        st.write("Output Tensor Shape:", output.shape)

        # displaying the queries, keys, and values tensors
        st.write("Queries Tensor:")
        st.table(pd.DataFrame(attention_layer.queries[0].detach().numpy()))
        st.write("Keys Tensor:")
        st.table(pd.DataFrame(attention_layer.keys[0].detach().numpy()))
        st.write("Values Tensor:")
        st.table(pd.DataFrame(attention_layer.values[0].detach().numpy()))
