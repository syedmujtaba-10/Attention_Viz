import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import PIL.Image
import seaborn as sns
import networkx as nx
from wordcloud import WordCloud
import shap

def plot_interactive_attention(attention, tokens, layer=0, head=0):
    """Creates an interactive heatmap using Plotly"""
    attn_weights = attention[layer][0, head].detach().numpy()
    df = pd.DataFrame(attn_weights, index=tokens, columns=tokens)

    fig = px.imshow(df, labels=dict(x="Token", y="Token", color="Attention"))
    return fig




def plot_attention_heatmap(attention, tokens, layer=0, head=0):
    """Creates an interactive attention heatmap and returns a Plotly figure."""
    attn_weights = attention[layer][0, head].detach().numpy()

    # Convert attention weights to DataFrame
    df = pd.DataFrame(attn_weights, index=tokens, columns=tokens)

    # Generate the heatmap using Plotly
    fig = px.imshow(df, labels=dict(x="Token", y="Token", color="Attention"),
                    x=tokens, y=tokens, color_continuous_scale="Blues")

    fig.update_layout(title=f"Attention Heatmap (Layer {layer}, Head {head})")

    return fig  # ✅ Now returns a Plotly figure




def plot_attention_graph(attention, tokens, layer=0, head=0):
    """Creates a force-directed graph of attention weights."""
    attn_weights = attention[layer][0, head].detach().numpy()
    
    G = nx.DiGraph()
    
    # Add nodes (tokens)
    for i, token in enumerate(tokens):
        G.add_node(i, label=token)

    # Add weighted edges
    edges = []
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            weight = float(attn_weights[i, j])  # ✅ Convert NumPy float32 to Python float
            if weight > 0.1:  # Filter weak attention links
                G.add_edge(i, j, weight=weight)
                edges.append(weight)

    # Get positions using force layout
    pos = nx.spring_layout(G, seed=42, k=0.5)

    # Create Plotly graph
    edge_x, edge_y = [], []
    edge_width = []  # Stores float values for line width

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_width.append(float(edge[2]["weight"]) * 5)  # ✅ Ensure Python float

    # Draw edges
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, 
        line=dict(width=max(edge_width, default=1), color="blue"),  # ✅ Use max width if empty
        mode="lines"
    )

    # Draw nodes
    node_x, node_y, node_text = [], [], []
    for node, (x, y) in pos.items():
        node_x.append(x)
        node_y.append(y)
        node_text.append(tokens[node])

    node_trace = go.Scatter(
        x=node_x, y=node_y, 
        mode="markers+text", text=node_text,
        textposition="top center",
        marker=dict(size=10, color="red")
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(title=f"Attention Graph (Layer {layer}, Head {head})", showlegend=False)
    return fig



def plot_attention_wordcloud(attention, tokens, layer=0, head=0):
    """Generates a word cloud where size represents attention importance and returns a Plotly figure."""
    attn_weights = attention[layer][0, head].detach().numpy()
    avg_attn = np.mean(attn_weights, axis=0)  # Aggregate attention per token

    # Create dictionary for word cloud
    word_freq = {tokens[i]: avg_attn[i] * 100 for i in range(len(tokens))}

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color="white",
                          colormap="Blues").generate_from_frequencies(word_freq)

    # Convert Matplotlib word cloud to image
    img_buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(img_buffer, format="png", bbox_inches="tight")
    plt.close()

    # Convert image buffer to a PIL image and then to a Plotly figure
    img = PIL.Image.open(img_buffer)
    fig = px.imshow(img)
    fig.update_layout(title=f"Attention Word Cloud (Layer {layer}, Head {head})", xaxis=dict(visible=False), yaxis=dict(visible=False))

    return fig  # ✅ Now returns a Plotly figure




def plot_attention_arcs(attention, tokens, layer=0, head=0):
    """Visualizes attention as an arc diagram and returns a Plotly figure."""
    attn_weights = attention[layer][0, head].detach().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, len(tokens) - 1)
    ax.set_ylim(0, 2)
    
    # Draw tokens as points
    for i, token in enumerate(tokens):
        ax.text(i, 0, token, ha="center", fontsize=12, rotation=45)

    # Draw arcs for strong attention links
    for i in range(len(tokens)):
        for j in range(len(tokens)):
            weight = attn_weights[i, j]
            if weight > 0.1:  # Only draw strong attention links
                arc_height = weight * 2  # Higher weight = bigger arc
                arc = plt.Circle(((i + j) / 2, 0), arc_height, fill=False, edgecolor="blue", linewidth=weight * 5)
                ax.add_patch(arc)
    
    ax.set_axis_off()
    plt.title(f"Attention Arc Diagram (Layer {layer}, Head {head})")

    # Convert Matplotlib figure to Plotly-compatible image
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight")
    plt.close()
    
    # Convert image buffer to a PIL image and then to a Plotly figure
    img = PIL.Image.open(img_buffer)
    fig = go.Figure()
    fig.add_trace(go.Image(z=img))
    fig.update_layout(title=f"Attention Arc Diagram (Layer {layer}, Head {head})",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))

    return fig  # ✅ Now returns a Plotly figure



import streamlit as st

def highlight_attention(text, attention, tokens, layer=0, head=0):
    """Highlights tokens based on attention strength."""
    attn_weights = attention[layer][0, head].detach().numpy()
    avg_attn = np.mean(attn_weights, axis=0)  # Compute average attention per token

    # Normalize attention values to range [0, 1]
    max_attn = np.max(avg_attn)
    min_attn = np.min(avg_attn)
    normalized_attn = (avg_attn - min_attn) / (max_attn - min_attn + 1e-8)

    # Generate HTML spans with color intensity
    highlighted_text = " ".join(
        f'<span style="background-color: rgba(255, 0, 0, {normalized_attn[i]}); padding:2px 4px; border-radius:4px;">{tokens[i]}</span>'
        for i in range(len(tokens))
    )
    
    return f"<p style='font-size:18px;'>{highlighted_text}</p>"



def plot_attention_trend(attention, tokens, head=0):
    """Plots attention strength across all layers for a given head."""
    avg_attn_per_layer = [np.mean(layer[0, head].detach().numpy()) for layer in attention]
    layers = list(range(len(attention)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=layers, y=avg_attn_per_layer, mode="lines+markers", name="Attention Strength"))

    fig.update_layout(title="Attention Strength Across Layers",
                      xaxis_title="Layer",
                      yaxis_title="Average Attention Weight",
                      template="plotly_dark")

    return fig


import shap
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from io import BytesIO
import PIL.Image
import torch

def explain_attention_with_shap(model, tokenizer, text):
    """Computes SHAP explanations for transformer attention and returns a summary plot image."""

    # Ensure input text is wrapped in a list (SHAP sometimes passes numpy arrays)
    if isinstance(text, np.ndarray):
        text = text.tolist()  # Convert NumPy array to Python list
    if isinstance(text, str):
        text = [text]  # Ensure single text input is a list

    # Define function for SHAP to explain
    def model_forward(text_list):
        """Tokenizes text, converts to tensor, and gets model output."""
        if isinstance(text_list, np.ndarray):
            text_list = text_list.tolist()  # Convert NumPy array to Python list
        elif isinstance(text_list, str):
            text_list = [text_list]  # Ensure text is wrapped in a list

        # Tokenize input correctly
        tokenized_inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True)
        input_ids = tokenized_inputs["input_ids"]
        attention_mask = tokenized_inputs["attention_mask"]

        with torch.no_grad():  # Disable gradient computation for inference
            output = model(input_ids, attention_mask=attention_mask)
        
        # Extract meaningful output (last hidden state or attention weights)
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state[:, 0, :].cpu().numpy()  # Extract CLS token representation
        elif hasattr(output, "attentions"):
            return torch.mean(output.attentions[-1], dim=1).cpu().numpy()  # Extract average attention
        else:
            raise ValueError("Model output does not contain `last_hidden_state` or `attentions`.")

    # Define SHAP explainer (now properly handling text input)
    explainer = shap.Explainer(model_forward, masker=shap.maskers.Text(tokenizer))

    # Compute SHAP values
    shap_values = explainer(text)

    # Create SHAP summary plot
    fig, ax = plt.subplots(figsize=(10, 5))
    shap.plots.text(shap_values, display=False)
    plt.title("SHAP Explanation for Attention")

    # Convert Matplotlib figure to an image for Streamlit
    img_buffer = BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight")
    plt.close()

    img = PIL.Image.open(img_buffer)
    return img
