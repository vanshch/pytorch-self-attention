import streamlit as st
import matplotlib.pyplot as plt
from attention import Attention, SampleTokenizer
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

# button to trigger sentence processing
if st.button("Process Sentence"):
    tokenizer = SampleTokenizer(user_sentence, hidden_dim)
    attention_layer = Attention(dim_in=hidden_dim, dim_out=hidden_dim)
    output = tokenizer.embedded.unsqueeze(0)  
    attention_score = attention_layer.forward(tokenizer.embedded.unsqueeze(0))

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
    cax = ax.matshow(attention_score[0].detach().numpy(), cmap="viridis") 
    fig.colorbar(cax) 
    ax.set_xticks(range(len(tokenizer.encoded)))
    ax.set_yticks(range(len(tokenizer.encoded)))
    ax.set_xticklabels(user_sentence.split(' '), rotation=90)
    ax.set_yticklabels(user_sentence.split(' '))
    st.pyplot(fig)
    st.write("Output Tensor Shape:", output.shape)

    # displaying the queries, keys, and values tensors
    st.write("Queries Tensor:")
    st.table(pd.DataFrame(attention_layer.queries[0].detach().numpy())) 
    st.write("Keys Tensor:")
    st.table(pd.DataFrame(attention_layer.keys[0].detach().numpy()))
    st.write("Values Tensor:")
    st.table(pd.DataFrame(attention_layer.values[0].detach().numpy()))




    
