import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

# PHQ9 question scores: integers 0-3
phq_scores = {f'phq{i}': np.random.randint(0, 4, size=n) for i in range(1, 10)}

# Age: 18-65, skewed normal distribution
ages = np.random.normal(loc=35, scale=10, size=n).astype(int)
ages = np.clip(ages, 18, 65)

# Gender: weighted random choice
genders = np.random.choice(['male', 'female', 'non-binary'], size=n, p=[0.45, 0.45, 0.10])

# Happiness score: float 0-10 with some noise
happiness = np.random.normal(loc=5, scale=2, size=n)
happiness = np.clip(happiness, 0, 10).round(2)

# Period name: random choice with some distribution
periods = np.random.choice(['morning', 'midday', 'evening'], size=n, p=[0.3, 0.4, 0.3])

# Combine into dataframe
df = pd.DataFrame(phq_scores)
df['age'] = ages
df['gender'] = genders
df['happiness.score'] = happiness
df['period.name'] = periods

df.to_csv('synthetic_phq9_dataset.csv', index=False)
print("Dataset generated with shape:", df.shape)
