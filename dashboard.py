import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config for a dashboard look
st.set_page_config(page_title="Wild Animal Detection Dashboard", layout="wide")

# Load data from results.csv (if available)
def load_results():
    try:
        df = pd.read_csv("results.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=["Date", "Time", "Species", "Accuracy"])

# Display the results in a table
def display_results_table():
    df = load_results()
    if df.empty:
        st.warning("No results found!")
    else:
        st.dataframe(df)

# Plot species counts as a bar chart
def plot_species_counts():
    df = load_results()
    if not df.empty:
        if 'Species' in df.columns:  # Check if 'Species' column exists
            species_count = df['Species'].value_counts()
            st.subheader("Number of Predictions by Species")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=species_count.values, y=species_count.index, ax=ax)
            ax.set_xlabel('Species')
            ax.set_ylabel('Count')
            ax.set_title('Prediction Count for Each Species')
            st.pyplot(fig)
        else:
            st.warning("'Species' column not found in results.csv")
    else:
        st.warning("No results available to plot.")

# Provide download button for CSV
def download_results():
    df = load_results()
    if not df.empty:
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Results as CSV",
            data=csv,
            file_name="results.csv",
            mime="text/csv"
        )

# Main Dashboard layout
def main_dashboard():
    st.title("🐾 Wild Animal Detection Dashboard")
    st.write("Here you can view and download the results of the animal species predictions.")
    
    display_results_table()  # Display the results table
    plot_species_counts()    # Plot species counts bar chart
    download_results()       # Provide CSV download option

# Run the dashboard
if __name__ == "__main__":
    main_dashboard()
