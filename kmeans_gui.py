import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from kmeans_logic import process_data

class KMeansGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("K-Means Clustering")
        self.geometry("800x600")  # Set a larger window size

        # Add widgets for file upload, data percentage, number of clusters, and results
        self.create_widgets()

    def create_widgets(self):
        # File Upload
        self.file_label = tk.Label(self, text="Select a CSV file:")
        self.file_label.pack(pady=10)

        self.upload_button = tk.Button(self, text="Upload CSV", command=self.upload_file)
        self.upload_button.pack(pady=10)

        self.file_path = tk.StringVar()
        self.file_path_label = tk.Label(self, textvariable=self.file_path)
        self.file_path_label.pack(pady=5)

        # Percentage Input
        self.percentage_label = tk.Label(self, text="Percentage of Data to Process (e.g., 80):")
        self.percentage_label.pack(pady=10)

        self.percentage_entry = tk.Entry(self)
        self.percentage_entry.pack(pady=5)

        # Number of Clusters Input
        self.num_clusters_label = tk.Label(self, text="Number of Clusters:")
        self.num_clusters_label.pack(pady=10)

        self.num_clusters_entry = tk.Entry(self)
        self.num_clusters_entry.pack(pady=5)

        # Process Button
        self.process_button = tk.Button(self, text="Run K-Means", command=self.process_data)
        self.process_button.pack(pady=20)

        # Results Display Area
        # Results Display Area
        self.results_text = tk.Text(self, width=100, height=100)  # Increase width and height
        self.results_text.pack(pady=10)


    def upload_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.file_path.set(file_path)

    def process_data(self):
        file_path = self.file_path.get()
        percentage = float(self.percentage_entry.get())
        num_clusters = int(self.num_clusters_entry.get())

        if not file_path or percentage <= 0 or num_clusters <= 0:
            messagebox.showerror("Invalid Input", "Please provide valid inputs for file, percentage, and number of clusters.")
            return

        try:
            df, outliers, cluster_labels, final_centroids = process_data(file_path, percentage, num_clusters)
            self.display_results(df, outliers, num_clusters)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def display_results(self, df, outliers, num_clusters):
        self.results_text.delete(1.0, tk.END)  # Clear previous results

        # Display clusters
        for cluster_id in range(num_clusters):
            self.results_text.insert(tk.END, f"\nCluster {cluster_id}:\n")
            cluster_data = df[df['Cluster'] == cluster_id]
            self.results_text.insert(tk.END, cluster_data[['CustomerID', 'Gender','Age', 'Annual Income (k$)', 'Spending Score (1-100)']].to_string(index=False))
        
        # Display outliers
        if outliers:
            self.results_text.insert(tk.END, "\nOutliers:\n")
            outlier_df = df.iloc[outliers]
            self.results_text.insert(tk.END, outlier_df[['CustomerID', 'Gender','Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']].to_string(index=False))
        else:
            self.results_text.insert(tk.END, "\nNo outliers detected.\n")

if __name__ == "__main__":
    app = KMeansGUI()
    app.mainloop()
