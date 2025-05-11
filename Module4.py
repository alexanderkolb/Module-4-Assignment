import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#Read in and clean dataset
df = pd.read_csv('nba_team_stats_00_to_23.csv', low_memory = False)
df = df.drop(df[df.season != "2023-24"].index)
df = df.dropna()

##Chosing columns we will be using for this analysis
relevant_columns = ['win_percentage', 'points', 'rebounds', 'assists', 'field_goals_attempted', 'field_goal_percentage', 
                'three_pointers_attempted', 'three_point_percentage', 'free_throw_attempted', 'free_throw_percentage', 
                'turnovers', 'steals', 'blocks']

df_cluster = df[relevant_columns].copy()
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_cluster)

inertia = []
k_values = range(1, 11)
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(df_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares')
plt.title('Elbow Method for Optimal k')
plt.grid(True)
plt.show()

k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

for i in range(k):
    print(f"\nCluster {i}:")
    group = df[df["cluster"] == i]
    print(group[['Team', 'win_percentage', 'three_point_percentage', 'field_goal_percentage', 'assists', 'steals', 'blocks', 'plus_minus']])

    
