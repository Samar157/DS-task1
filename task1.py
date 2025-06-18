import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample data
data = pd.DataFrame({
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Other', 'Female', 'Male', 'Male', 'Other', 'Female']
})

# Count of each gender
gender_counts = data['Gender'].value_counts()

# Bar chart
sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='pastel')
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()

import numpy as np

# Sample data
np.random.seed(42)
ages = np.random.normal(loc=35, scale=10, size=100)  # mean age 35, std 10

# Histogram
import numpy as np

# Sample data
np.random.seed(42)
ages = np.random.normal(loc=35, scale=10, size=100)  # mean age 35, std 10

# Histogram
sns.histplot(ages, bins=10, kde=True, color='skyblue')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

