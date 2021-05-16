# -*- coding: utf-8 -*-
"""
Created on Sat May 15 20:36:43 2021

@author: Thomas Wortmann
"""
import pandas as pd
import seaborn as sn # You might need to install seaborn with pip install seaborn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def correlationMatrix(method):
    # Correlation Matrix
    assert method == "pearson" or method == "kendall" or method == "spearman", "Method should be pearson, kendall, or spearman"
        
    corrMatrix = indicators.corr(method = method)
    print(f"Correlation {method}")
    print (corrMatrix)
    print("\n")
    
    # Figure
    figDPI = 300
    figSize = (5, 4)
    
    plt.figure(figsize = figSize) # New figure
    sn.heatmap(corrMatrix, annot=True)
    
    plt.savefig(path + f'correlation_{method}.png', dpi = figDPI)
    plt.show()
    
def principalComponentsAnalysis(dataframe, columns):
    #Run PCA
    pca = PCA(n_components=len(columns))
    pca.fit(dataframe[columns])

    # Format table with results
    principal_components = pd.DataFrame.from_records(pca.components_)
    principal_components.columns = columns
    principal_components["Explained Variance Ratio"] = pca.explained_variance_ratio_
    
    print(f"PCA")
    print(principal_components)
    print("\n")
    
    # Export as EXCEL file
    principal_components.to_excel(f"{path}pca_{'_'.join(columns)}.xlsx", index = False)
    
    
# Use double spaces on Windows
path = "D:\\Sync\\Academic\\Teaching\\Uni Stuttgart\\2021_SS\\Computational Explorations\\Git\\ComputationalExplorations2021\\ML\\"
file = "unsupervised_test_data.csv"

# Read the data with Pandas (sep is the seperator for the csv file)
samples = pd.read_csv(path + file, sep=',')

# Choose the relevant columns from the data and rename them
indicators = samples[["out:EEI", "out:NEI", "out:FE", "out:SE", "out:FAR", "out:OSI", "out:RE", ]]
indicators.columns = ["EEI", "NEI", "FE", "SE", "FAR", "OSI", "RE"]

# Print the first three rows to check
print("Data to analyze")
print(indicators.iloc[: 3])
print("\n")

# Print Correlation Matrices
correlationMatrix("pearson")
correlationMatrix("spearman")
correlationMatrix("kendall")

# Principal Components Analysis

# Columns to include in analysis
solar_columns = ["EEI", "NEI", "FE", "SE"]
shade_columns = ["FAR", "OSI"]
solar_indicators = principalComponentsAnalysis(indicators, solar_columns)
shade_indicators = principalComponentsAnalysis(indicators, shade_columns)



