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

    fig, ax = plt.subplots(figsize=(6, 3))
    fig = px.imshow(df, labels=dict(x="Token", y="Token", color="Attention"))
    return fig




def plot_attention_heatmap(attention, tokens, layer=0, head=0):
    """Creates an interactive attention heatmap and returns a Plotly figure."""
    attn_weights = attention[layer][0, head].detach().numpy()

    # Convert attention weights to DataFrame
    df = pd.DataFrame(attn_weights, index=tokens, columns=tokens)

    # Generate the heatmap using Plotly
    fig, ax = plt.subplots(figsize=(6, 3))
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

    fig, ax = plt.subplots(figsize=(6, 3))
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
    plt.figure(figsize=(6, 3))
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
    
    fig, ax = plt.subplots(figsize=(6, 3))
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
    print(attention[layer][0, head].detach().numpy())
    print(tokens)
    
    return f"<p style='font-size:18px;'>{highlighted_text}</p>"



def plot_attention_trend(
    attention, 
    tokens, 
    head=0, 
    special_tokens={"[CLS]", "[SEP]", ".", ",", "or"}, 
    direction="to"
):
    """
    Plots how much each special token is attended to or attends outward across all layers
    for a given head.

    Parameters:
    -----------
    attention : list of torch.Tensor
        A list of attention tensors across layers. Each element has shape:
        (1, num_heads, seq_len, seq_len)
    tokens : list of str
        The tokenized text from the model (including special tokens).
    head : int
        The specific attention head to examine.
    special_tokens : set of str
        Which tokens to track in the plot. Adjust for each model if needed.
    direction : str
        - "to"   => measure how much each token is *attended to* by the rest of the sequence
        - "from" => measure how much each token *attends outward* to the rest of the sequence
    """
    data = []
    num_layers = len(attention)

    for layer_idx in range(num_layers):
        # attn_weights shape after removing batch dimension => (num_heads, seq_len, seq_len)
        attn_weights = attention[layer_idx][0].detach().numpy()  
        # Extract the head we care about
        head_matrix = attn_weights[head]  # shape: (seq_len, seq_len)

        # For each token in the sequence
        for i, token_str in enumerate(tokens):
            if token_str in special_tokens:
                if direction == "to":
                    # How much token i is attended to by others => average over the query dimension
                    # i.e., mean of head_matrix[:, i]
                    value = np.mean(head_matrix[:, i])
                else:
                    # direction == "from"
                    # How much token i attends outward => average over the key dimension
                    # i.e., mean of head_matrix[i, :]
                    value = np.mean(head_matrix[i, :])

                data.append({
                    "Layer": layer_idx + 1,   # 1-based indexing for readability
                    "Token": token_str,
                    "Attention": value
                })

    # Convert to DataFrame for plotting
    df = pd.DataFrame(data)

    # Create a Plotly figure
    fig = go.Figure()

    # Plot one line per special token
    for token_name in df["Token"].unique():
        subset = df[df["Token"] == token_name]
        fig.add_trace(go.Scatter(
            x=subset["Layer"],
            y=subset["Attention"],
            mode="lines+markers",
            name=token_name
        ))

    direction_label = "Attended To" if direction == "to" else "Attends Outward"
    fig.update_layout(
        title=f"Attention Trend Across Layers (Head {head}, {direction_label})",
        xaxis_title="Layer",
        yaxis_title="Average Attention",
        template="plotly_dark"
    )

    return fig




"""def plot_attention_by_layer(attention, tokens):
    
    #Plots how much each special token is attended to across layers.
    #Specifically, for each token j, we compute average attention from
    #all heads and all query positions to j.
    #
    layers = len(attention)  # e.g. 12 for BERT Base
    data = []

    # Adjust these special tokens if you use GPT-2 or RoBERTa
    special_tokens = {"[CLS]", "[SEP]", ".", ",", "or"}

    for layer in range(layers):
        # attn_weights shape: (num_heads, query_seq_len, key_seq_len)
        attn_weights = attention[layer][0].detach().numpy()

        # 1) Average over heads (axis=0)
        # 2) Average over query positions (axis=1)
        # => shape: (key_seq_len,)
        avg_attn_to = np.mean(attn_weights, axis=(0, 1))

        for i, token in enumerate(tokens):
            token_str = str(token)
            if token_str in special_tokens:
                data.append({
                    "Layer": layer + 1,
                    "Token": token_str,
                    "Attention": avg_attn_to[i]
                })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Plot using Seaborn
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x="Layer", y="Attention", hue="Token", style="Token", ax=ax)
    sns.lineplot(data=df, x="Layer", y="Attention", hue="Token", ax=ax, legend=False)

    plt.xlabel("Layer")
    plt.ylabel("Avg. Attention (Token is Attended To)")
    plt.title("Attention Scores Across Layers (Attended-To Metric)")
    plt.xticks(range(1, layers + 1))
    plt.grid(True)

    return fig"""
