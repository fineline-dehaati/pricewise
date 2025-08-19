import os, pandas as pd, streamlit as st
from router import route_query
from analytics.features import coerce_schema, basic_profile, normalize_schema, detect_column_types, clean_dataframe_for_display
from commentary import narrate_insights

# Configure matplotlib for Streamlit
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.ioff()  # Turn off interactive mode

st.set_page_config(page_title="Price Analytics Copilot", layout="wide")
st.title("📈 Price Analytics Copilot (LLM-enabled)")

with st.sidebar:
    show_code = st.checkbox("Show generated code", value=False)
    use_sample = st.checkbox("Load sample dataset", value=True)
    
    # Manual column override
    if hasattr(st.session_state, 'df') and st.session_state.df is not None:
        st.subheader("🔧 Column Override")
        st.caption("Manually adjust column mapping if needed")
        
        if hasattr(st.session_state, 'column_mapping'):
            original_cols = list(st.session_state.df.columns)
            
            # Date column
            date_col = st.selectbox(
                "Date Column", 
                original_cols, 
                index=original_cols.index(st.session_state.column_mapping.get('datetime', original_cols[0])) if st.session_state.column_mapping.get('datetime') in original_cols else 0
            )
            
            # ID column
            id_col = st.selectbox(
                "Location ID Column", 
                original_cols, 
                index=original_cols.index(st.session_state.column_mapping.get('Location_Id', original_cols[0])) if st.session_state.column_mapping.get('Location_Id') in original_cols else 0
            )
            
            # Name column
            name_col = st.selectbox(
                "Location Name Column", 
                original_cols, 
                index=original_cols.index(st.session_state.column_mapping.get('Location_Name', original_cols[0])) if st.session_state.column_mapping.get('Location_Name') in original_cols else 0
            )
            
            # Price column
            price_col = st.selectbox(
                "Price Column", 
                original_cols, 
                index=original_cols.index(st.session_state.column_mapping.get('price', original_cols[0])) if st.session_state.column_mapping.get('price') in original_cols else 0
            )
            
            # Apply manual override
            if st.button("Apply Column Mapping"):
                manual_mapping = {
                    'datetime': date_col,
                    'Location_Id': id_col,
                    'Location_Name': name_col,
                    'price': price_col
                }
                try:
                    normalized_df, _ = normalize_schema(st.session_state.df, manual_mapping)
                    st.session_state.df = normalized_df
                    st.session_state.column_mapping = manual_mapping
                    st.success("✅ Column mapping updated!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

uploaded = st.file_uploader(
    "Upload Price Data File", 
    type=["csv", "xlsx", "xls"],
    help="Upload CSV, Excel, or Parquet files. The app will automatically detect date, ID, name, and price columns."
)

if "df" not in st.session_state:
    st.session_state.df = None

if st.session_state.df is None and use_sample and uploaded is None:
    sample_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_prices.csv")
    if os.path.exists(sample_path):
        st.session_state.df = coerce_schema(pd.read_csv(sample_path))

if uploaded is not None:
    df = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded)
    # Use dynamic column detection
    try:
        normalized_df, column_mapping = normalize_schema(df)
        st.session_state.df = normalized_df
        st.session_state.column_mapping = column_mapping
        st.success(f"✅ Column mapping detected: {column_mapping}")
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        st.stop()

df = st.session_state.df
if df is None:
    st.info("Upload a dataset or enable sample data in the sidebar.")
    st.stop()

st.subheader("Dataset preview")

# Show column mapping if available
if hasattr(st.session_state, 'column_mapping') and st.session_state.column_mapping:
    # Convert internal names to user-friendly names for display
    display_mapping = {}
    for internal_name, original_name in st.session_state.column_mapping.items():
        if internal_name == 'datetime':
            display_mapping['📅 Date/Time'] = original_name
        elif internal_name == 'Location_Id':
            display_mapping['🆔 Location ID'] = original_name
        elif internal_name == 'Location_Name':
            display_mapping['📍 Location Name'] = original_name
        elif internal_name == 'price':
            display_mapping['💰 Price'] = original_name
        else:
            display_mapping[internal_name] = original_name
    
    st.info(f"📊 **Detected Columns:** {display_mapping}")

st.dataframe(df.head(50))
with st.expander("Quick profile"):
    st.json(basic_profile(df))

# Clean DataFrame for display to prevent PyArrow issues
df_clean = clean_dataframe_for_display(df)

question = st.chat_input("Ask a question about your prices…")
if question:
    with st.spinner("Working..."):
        plan, code, result, figs, warnings = route_query(df, question)
    
    st.markdown("### Answer")
    
    # Handle different result types with better error handling
    if result is not None:
        try:
            if hasattr(result, 'head') and callable(getattr(result, 'head', None)):
                # It's a DataFrame-like object
                try:
                    st.dataframe(result.head(100))
                except Exception as e:
                    st.error(f"❌ Error displaying DataFrame: {str(e)}")
                    # Fallback: show as text
                    st.text(str(result.head(10)))
            elif hasattr(result, 'figure'):
                # It's a matplotlib Axes object
                st.pyplot(result.figure)
            elif hasattr(result, 'get_figure'):
                # It's a matplotlib Axes object with get_figure method
                st.pyplot(result.get_figure())
            elif hasattr(result, 'axes'):
                # It's a matplotlib Figure object
                st.pyplot(result)
            else:
                # It's something else, try to display it
                st.write(result)
        except Exception as e:
            st.error(f"❌ Error displaying result: {str(e)}")
            st.text(f"Raw result: {str(result)[:500]}...")
    
    # Display any additional figures with better error handling
    if figs:
        st.markdown("### 📊 Generated Charts")
        for i, fig in enumerate(figs):
            try:
                if fig is not None:
                    st.pyplot(fig)
                    st.caption(f"Chart {i+1}")
                else:
                    st.warning(f"Chart {i+1} is None")
            except Exception as e:
                st.error(f"❌ Error displaying chart {i+1}: {str(e)}")
    else:
        st.info("💡 No charts were generated for this query. Try asking for visualizations like 'Show me a price trend chart' or 'Create a graph of price by location'")
    
    if warnings:
        st.warning("\n".join(warnings))
    
    if show_code and code:
        st.code(code, language="python")
    
    # Commentary
    st.markdown("### Commentary")
    st.markdown(narrate_insights(question, result, warnings, plan=plan))
