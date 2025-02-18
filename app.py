import streamlit as st
from model import get_attention, MODEL_OPTIONS, load_model
from visualize import (
    plot_interactive_attention, 
    plot_attention_heatmap, 
    plot_attention_graph, 
    plot_attention_wordcloud, 
    plot_attention_arcs,
    highlight_attention,
    plot_attention_trend,

)

st.title("🧠 Compare Attention Across Transformer Models")

# Select two models to compare
col1, col2 = st.columns(2)

with col1:
    model1 = st.selectbox("Choose First Model", list(MODEL_OPTIONS.keys()), key="model1")

with col2:
    model2 = st.selectbox("Choose Second Model", list(MODEL_OPTIONS.keys()), key="model2")

# User input for text, layer, and head selection
text = st.text_area("Enter Text", "The quick brown fox jumps over the lazy dog.")
layer = st.slider("Select Layer", 0, 11, 0)  # Adjust based on model architecture
head = st.slider("Select Head", 0, 11, 0)

if st.button("Compare Models"):
    with st.spinner("Processing..."):
        selected_model1 = MODEL_OPTIONS[model1]
        selected_model2 = MODEL_OPTIONS[model2]

        attention1, tokens1 = get_attention(text, selected_model1)
        attention2, tokens2 = get_attention(text, selected_model2)

        st.subheader(f"📊 Comparing {model1} vs. {model2}")


        # Show attention-highlighted text for both models
        st.subheader(f"🔎 Highlighted Attention in Text (Layer {layer}, Head {head})")
        st.caption("Highlights tokens based on attention strength")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{model1}**", unsafe_allow_html=True)
            highlighted_text1 = highlight_attention(text, attention1, tokens1, layer, head)
            st.markdown(highlighted_text1, unsafe_allow_html=True)

        with col2:
            st.markdown(f"**{model2}**", unsafe_allow_html=True)
            highlighted_text2 = highlight_attention(text, attention2, tokens2, layer, head)
            st.markdown(highlighted_text2, unsafe_allow_html=True)

        # Show attention heatmaps
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"🔍 {model1} Attention Heatmap")
            fig1 = plot_attention_heatmap(attention1, tokens1, layer, head)
            st.plotly_chart(fig1)

        with col2:
            st.subheader(f"🔍 {model2} Attention Heatmap")
            fig2 = plot_attention_heatmap(attention2, tokens2, layer, head)
            st.plotly_chart(fig2)

        # Show attention word clouds
        
        st.subheader("Word Cloud")
        st.caption("Generates a word cloud where size represents attention importance")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"📌 {model1} Word Cloud")
            fig3 = plot_attention_wordcloud(attention1, tokens1, layer, head)
            st.plotly_chart(fig3)

        with col2:
            st.subheader(f"📌 {model2} Word Cloud")
            fig4 = plot_attention_wordcloud(attention2, tokens2, layer, head)
            st.plotly_chart(fig4)

        # Show interactive attention graphs
        st.subheader("Attention Graph")
        st.caption("Creates a force-directed graph of attention weights")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"🔗 {model1} Attention Graph")
            fig5 = plot_attention_graph(attention1, tokens1, layer, head)
            st.plotly_chart(fig5)

        with col2:
            st.subheader(f"🔗 {model2} Attention Graph")
            fig6 = plot_attention_graph(attention2, tokens2, layer, head)
            st.plotly_chart(fig6)

        # Show attention arc diagrams
        st.subheader("Attention Arc Diagram")
        st.caption("Visualizes attention as an arc diagram and returns a Plotly figure")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"🌀 {model1} Attention Arc Diagram")
            fig7 = plot_attention_arcs(attention1, tokens1, layer, head)
            st.plotly_chart(fig7)

        with col2:
            st.subheader(f"🌀 {model2} Attention Arc Diagram")
            fig8 = plot_attention_arcs(attention2, tokens2, layer, head)
            st.plotly_chart(fig8)

        # Show Attention Trends Over Layers
        st.subheader("📈 Attention Trend for Special Tokens")
        st.caption("Plots how much each special token is attended to or attends outward across all layers for a given head.")
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"📊 {model1} Attention Trend")
            fig9 = plot_attention_trend(attention1, tokens1, head)
            st.plotly_chart(fig9)

        with col2:
            st.subheader(f"📊 {model2} Attention Trend")
            fig10 = plot_attention_trend(attention2, tokens2, head)
            st.plotly_chart(fig10)
