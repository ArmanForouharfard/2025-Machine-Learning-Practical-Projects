# ============================================================
#Machine Learning / Regression Project
#Yadegar-Imam University
#Dr S. Abolfazl Hosseini
#Gathered by Arman Forouharfard using Chat-GPT 4 & DeepSeek
# Year: 2025

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans

class TwoMoonKMeansApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Two Moon K-Means Clustering")
        self.master.geometry("900x700")
        
        # Variables
        self.X = None
        self.y = None
        self.kmeans = None
        self.current_k = 2
        self.random_state = 42
        
        # Create UI
        self.create_widgets()
        self.generate_data()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Number of clusters slider
        ttk.Label(control_frame, text="Number of Clusters (k):").pack(pady=(10, 0))
        self.k_slider = ttk.Scale(control_frame, from_=1, to=10, value=2, 
                                 command=self.update_k_value)
        self.k_slider.pack(fill=tk.X, padx=5, pady=5)
        
        self.k_value = tk.StringVar(value="2")
        ttk.Label(control_frame, textvariable=self.k_value).pack()
        
        # Regenerate data button
        ttk.Button(control_frame, text="Regenerate Data", 
                  command=self.generate_data).pack(pady=10, fill=tk.X)
        
        # Apply K-means button
        ttk.Button(control_frame, text="Apply K-Means", 
                  command=self.apply_kmeans).pack(pady=5, fill=tk.X)
        
        # Performance info
        ttk.Label(control_frame, text="Performance:").pack(pady=(10, 0))
        self.perf_label = ttk.Label(control_frame, text="Not run yet")
        self.perf_label.pack()
        
        # Plot frame
        plot_frame = ttk.LabelFrame(main_frame, text="Visualization")
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_k_value(self, event=None):
        self.current_k = int(float(self.k_slider.get()))
        self.k_value.set(str(self.current_k))
        
    def generate_data(self):
        # Generate two moon data with some noise
        self.X, self.y = make_moons(n_samples=500, noise=0.08, random_state=self.random_state)
        self.random_state = np.random.randint(0, 1000)  # Change random state for next generation
        self.plot_data()
        
    def plot_data(self, labels=None, centers=None):
        self.ax.clear()
        
        if labels is None:
            # Plot original two moons
            self.ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='viridis', 
                           edgecolor='k', s=50, alpha=0.6)
            self.ax.set_title("Original Two Moon Data")
        else:
            # Plot clustered data
            self.ax.scatter(self.X[:, 0], self.X[:, 1], c=labels, cmap='viridis', 
                           edgecolor='k', s=50, alpha=0.6)
            
            if centers is not None:
                # Plot cluster centers
                self.ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', 
                               s=200, alpha=1, linewidths=2, edgecolor='k')
                self.ax.set_title(f"K-Means Clustering (k={self.current_k})")
        
        self.ax.grid(True)
        self.canvas.draw()
        
    def apply_kmeans(self):
        if self.X is None:
            messagebox.showerror("Error", "No data available!")
            return
            
        try:
            # Record start time for performance measurement
            import time
            start_time = time.time()
            
            # Apply K-means with random initialization
            self.kmeans = KMeans(n_clusters=self.current_k, init='random', n_init=10)
            labels = self.kmeans.fit_predict(self.X)
            
            # Calculate execution time
            exec_time = time.time() - start_time
            
            # Update performance label
            self.perf_label.config(text=f"Execution time: {exec_time:.4f} seconds\n"
                                      f"Inertia: {self.kmeans.inertia_:.2f}")
            
            # Plot results
            self.plot_data(labels=labels, centers=self.kmeans.cluster_centers_)
            
        except Exception as e:
            messagebox.showerror("Error", f"K-means failed: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TwoMoonKMeansApp(root)
    root.mainloop()