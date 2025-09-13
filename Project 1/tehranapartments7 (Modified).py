# ============================================================
#Machine Learning / Tehran Apartments Regression Project
#Yadegar-Imam University
#Dr S. Abolfazl Hosseini
#Gathered by Arman Forouharfard using Chat-GPT 4 & DeepSeek
# Year: 2025

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class TehranApartmentsApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Tehran Apartments Regression Analysis")
        self.master.geometry("1000x700")
        
        # Variables
        self.df = None
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        
        # Create UI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.master)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        control_frame = ttk.LabelFrame(main_frame, text="Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Load data button
        ttk.Button(control_frame, text="Load Dataset", command=self.load_data).pack(pady=5, fill=tk.X)
        
        # Feature selection
        ttk.Label(control_frame, text="Select Features:").pack(pady=(10,0), anchor=tk.W)
        self.feature_listbox = tk.Listbox(control_frame, selectmode=tk.MULTIPLE, height=5)
        self.feature_listbox.pack(fill=tk.X, padx=5, pady=5)
        
        # Target selection
        ttk.Label(control_frame, text="Select Target:").pack(pady=(10,0), anchor=tk.W)
        self.target_var = tk.StringVar()
        self.target_combobox = ttk.Combobox(control_frame, textvariable=self.target_var, state='readonly')
        self.target_combobox.pack(fill=tk.X, padx=5, pady=5)
        
        # Test size
        ttk.Label(control_frame, text="Test Size:").pack(pady=(10,0), anchor=tk.W)
        self.test_size = tk.DoubleVar(value=0.2)
        ttk.Entry(control_frame, textvariable=self.test_size).pack(fill=tk.X, padx=5, pady=5)
        
        # Plot selection
        ttk.Label(control_frame, text="Plot Type:").pack(pady=(10,0), anchor=tk.W)
        self.plot_type = tk.StringVar(value="scatter")
        ttk.Combobox(control_frame, textvariable=self.plot_type, 
                    values=["scatter", "heatmap", "countplot", "3D scatter", "poly_compare"]).pack(fill=tk.X, padx=5, pady=5)

        
        # Feature for plotting
        ttk.Label(control_frame, text="Feature for Plot:").pack(pady=(10,0), anchor=tk.W)
        self.plot_feature = tk.StringVar()
        self.plot_feature_combobox = ttk.Combobox(control_frame, textvariable=self.plot_feature, state='readonly')
        self.plot_feature_combobox.pack(fill=tk.X, padx=5, pady=5)
        
        # Run analysis button
        ttk.Button(control_frame, text="Run Analysis", command=self.run_analysis).pack(pady=10, fill=tk.X)
        
        # Right panel - Results
        result_frame = ttk.LabelFrame(main_frame, text="Results")
        result_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Notebook for tabs
        self.notebook = ttk.Notebook(result_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Stats tab
        self.stats_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_tab, text="Statistics")
        
        self.stats_text = tk.Text(self.stats_tab, wrap=tk.WORD)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Plot tab
        self.plot_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_tab, text="Visualization")
        
        # Model tab
        self.model_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.model_tab, text="Model")
        
        self.model_text = tk.Text(self.model_tab, wrap=tk.WORD)
        self.model_text.pack(fill=tk.BOTH, expand=True)
        
    def load_data(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.df.columns = self.df.columns.str.strip()
                
                # Update feature listbox
                self.feature_listbox.delete(0, tk.END)
                self.target_combobox['values'] = ()
                self.plot_feature_combobox['values'] = ()
                
                for col in self.df.columns:
                    self.feature_listbox.insert(tk.END, col)
                
                # Set default target to TotalPrice if exists
                if 'TotalPrice' in self.df.columns:
                    self.target_var.set('TotalPrice')
                else:
                    self.target_var.set(self.df.columns[-1])
                
                self.target_combobox['values'] = tuple(self.df.columns)
                self.plot_feature_combobox['values'] = tuple(self.df.columns)
                
                # Show basic stats
                self.show_stats()
                
                messagebox.showinfo("Success", "Dataset loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load file: {str(e)}")
    
    def show_stats(self):
        if self.df is not None:
            stats = self.df.describe().to_string()
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, stats)
    
    def run_analysis(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load a dataset first!")
            return

        selected_features = [self.feature_listbox.get(i) for i in self.feature_listbox.curselection()]
        target = self.target_var.get()

        if not selected_features or not target:
            messagebox.showerror("Error", "Please select at least one feature and a target!")
            return

        try:
            df_encoded = self.df.copy()

            # Encode categorical features
            for col in selected_features:
                if df_encoded[col].dtype == 'object':
                    df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))

            if df_encoded[target].dtype == 'object':
                df_encoded[target] = LabelEncoder().fit_transform(df_encoded[target].astype(str))

            X = df_encoded[selected_features]
            y = df_encoded[target]

            try:
                test_size = float(self.test_size.get())
                if not 0 < test_size < 1:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Error", "Test size must be a float between 0 and 1.")
                return

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            self.model = LinearRegression()
            self.model.fit(self.X_train, self.y_train)
            y_test_pred = self.model.predict(self.X_test)

            self.show_model_results(y_test_pred)
            self.create_plot()
        
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {str(e)}")
    
    def show_model_results(self, y_pred):
        if self.model is None:
            return
        
        # Calculate metrics
        r2 = self.model.score(self.X_test, self.y_test)
        coefficients = self.model.coef_
        intercept = self.model.intercept_
        
        # Format results
        result_text = f"Regression Results:\n\n"
        result_text += f"R-squared: {r2:.4f}\n"
        result_text += f"Intercept: {intercept:.2f}\n\n"
        result_text += "Coefficients:\n"
        
        for feature, coef in zip(self.X_train.columns, coefficients):
            result_text += f"{feature}: {coef:.4f}\n"
        
        # Update model tab
        self.model_text.delete(1.0, tk.END)
        self.model_text.insert(tk.END, result_text)
    
    def create_plot(self):
        for widget in self.plot_tab.winfo_children():
            widget.destroy()

        plot_type = self.plot_type.get()
        feature = self.plot_feature.get()
        y_pred = self.model.predict(self.X_test)

        if plot_type == "scatter" and feature:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(self.X_test[feature], self.y_test, color='red', label='Actual')
            ax.scatter(self.X_test[feature], y_pred, color='black', label='Predicted', alpha=0.6)
            sorted_indices = np.argsort(self.X_test[feature])
            ax.plot(self.X_test[feature].iloc[sorted_indices], y_pred[sorted_indices], color='blue', linewidth=2)
            ax.set_title('Actual vs Predicted Prices')
            ax.set_xlabel(feature)
            ax.set_ylabel(self.target_var.get())
            ax.legend()
            ax.grid(True)

        elif plot_type == "heatmap":
            numeric_cols = self.df.select_dtypes(include=np.number).columns
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(self.df[numeric_cols].corr(), annot=True, ax=ax)
            ax.set_title('Correlation Heatmap')

        elif plot_type == "countplot":
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.countplot(x=self.target_var.get(), data=self.df, ax=ax)
            ax.set_title('Count Plot')

        elif plot_type == "3D scatter":
            if len(self.X_test.columns) >= 2:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                x1 = self.X_test.iloc[:, 0]
                x2 = self.X_test.iloc[:, 1]
                ax.scatter(x1, x2, self.y_test, color='r', label='Actual')
                ax.scatter(x1, x2, y_pred, color='k', alpha=0.6, label='Predicted')
                sorted_indices = np.argsort(self.X_test[feature])
                ax.plot(self.X_test[feature].iloc[sorted_indices], y_pred[sorted_indices], color='green', linewidth=2)
                ax.set_xlabel(self.X_test.columns[0])
                ax.set_ylabel(self.X_test.columns[1])
                ax.set_zlabel(self.target_var.get())
                ax.set_title("3D Scatter Plot")
                ax.legend()

        elif plot_type == "poly_compare" and feature:
            fig, ax = plt.subplots(figsize=(8, 5))
            X_feature = self.X_train[[feature]].values
            y_feature = self.y_train.values

            # Plot actual data
            ax.scatter(X_feature, y_feature, color='gray', label='Actual Data')

            # Polynomial degrees to compare
            for degree in [2, 3, 4]:
                poly = PolynomialFeatures(degree)
                X_poly = poly.fit_transform(X_feature)
                model = LinearRegression()
                model.fit(X_poly, y_feature)
                X_range = np.linspace(X_feature.min(), X_feature.max(), 100).reshape(-1, 1)
                y_poly_pred = model.predict(poly.transform(X_range))
                ax.plot(X_range, y_poly_pred, label=f'Degree {degree}')
            
                ax.set_title(f'Polynomial Comparison ({feature})')
                ax.set_xlabel(feature)
                ax.set_ylabel(self.target_var.get())
                ax.legend()
                ax.grid(True)

        else:
            messagebox.showerror("Error", f"Plot type '{plot_type}' is not implemented correctly.")
            return

        # Embed plot
        canvas = FigureCanvasTkAgg(fig, master=self.plot_tab)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = TehranApartmentsApp(root)
    root.mainloop()