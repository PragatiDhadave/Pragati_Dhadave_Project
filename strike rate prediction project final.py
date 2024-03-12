#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import  RandomForestRegressor


# In[2]:


#Read a csv file
df = pd.read_csv('batting_summary.csv')
df


# In[3]:


import warnings
warnings.filterwarnings('ignore')


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.duplicated()


# In[7]:


df.duplicated().sum()


# In[8]:


df.shape


# In[9]:



df.describe()


# In[10]:


df.info()
print(df.size)


# In[11]:


df.isnull().sum()


# In[12]:


df.columns = df.columns.str.replace(" ","_")
df.columns


# In[13]:


df[df.Batsman_Names.isnull()]
df.Batsman_Names = df.Batsman_Names.fillna("Not Specified")


# In[14]:


df.rename(columns={'Out/Not_Out':'Out_or_Not'},inplace=True)


# In[15]:


# here we have dtypoe for numerical data is in object for runs scored 
df['Runs_Scored'].values


# In[16]:


# lets convert them into numeric 
df.Runs_Scored = pd.to_numeric(df['Runs_Scored'],errors="coerce")
df['Runs_Scored'].values


# In[17]:


df.info()


# In[18]:


df.Out_or_Not = df.Out_or_Not.fillna("not out") 
df.Runs_Scored = df.Runs_Scored.fillna(0.0)
df.Balls_Played = df.Balls_Played.fillna(1.0)
df.Fours = df.Fours.fillna(0)
df.Sixes = df.Sixes.fillna(0)
df.Strike_Rate = df.Strike_Rate.fillna(0)


# In[19]:


df.isnull().sum()


# In[20]:


print(df.columns)


# In[21]:


# lets do feature engineering for date and time
# convert Time into hours,day of week
df['Date'] = pd.to_datetime(df['Date'])

df['DayOfWeek'] = df['Date'].dt.dayofweek
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['Day'] = df['Date'].dt.day


# In[22]:


df.drop(columns=['Date', 'DayOfWeek'], inplace=True, errors='ignore')


# In[23]:


# lets take veiwe on first 5 record
df.head()


# # # Data Visualization
# 

# In[24]:


# importing libraries for data visualization:
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[25]:


# Best Strike Rate in 2023 and Top 5 Players


df_2023 = df[df['IPL_Edition'] == 2023]
player_strike_rate_2023 = df_2023.groupby('Batsman_Names')['Strike_Rate'].mean()

best_strike_rate_2023 = player_strike_rate_2023.idxmax()
print(f"\nPlayer with the Best Strike Rate in 2023: {best_strike_rate_2023}")

top5_strike_rate_2023 = player_strike_rate_2023.nlargest(5)
print("\nTop 5 Players with the Highest Strike Rate in 2023:")
print(top5_strike_rate_2023)


# In[26]:


plt.figure(figsize=(10, 6))
plt.barh(top5_strike_rate_2023.index, top5_strike_rate_2023.values, color='skyblue')
plt.xlabel('Strike Rate')
plt.ylabel('Player Names')
plt.title('Top 5 Players with the Highest Strike Rate in 2023')
plt.axvline(x=player_strike_rate_2023[best_strike_rate_2023], color='red', linestyle='--', label='Best Strike Rate')
plt.legend()
plt.show()


# In[27]:


# Year-wise batting-Strike Rate


yearly_strike_rate = df.groupby('IPL_Edition')['Strike_Rate'].mean()
print("Year-wise Average Strike Rate:")
print(yearly_strike_rate)


# In[28]:


# 2. Year-wise Strike Rate

yearly_strike_rate = df.groupby('IPL_Edition')['Strike_Rate'].mean()
# Convert the 'IPL_Edition' index to numeric for proper sorting
yearly_strike_rate.index = pd.to_numeric(yearly_strike_rate.index)
# Sort the data by IPL Edition
yearly_strike_rate = yearly_strike_rate.sort_index()
plt.figure(figsize=(10, 6))
plt.plot(yearly_strike_rate.index, yearly_strike_rate.values, marker='o', linestyle='-')
plt.title('Year-wise Average Strike Rate')
plt.xlabel('IPL Edition')
plt.ylabel('Average Strike Rate')
plt.grid(True)
plt.show()


# In[29]:


# most matches done in which stedium
#df.groupby('Stadium').count().max()[['Time']]
df.Stadium.value_counts()


# In[30]:


# Calculate the count of matches done in each stadium
stadium_counts_sorted = df['Stadium'].value_counts()

# Plot the count of matches done in each stadium using a line chart
plt.figure(figsize=(15, 10))
plt.plot(stadium_counts_sorted.index, stadium_counts_sorted.values, marker='o')
plt.title("Matches Done in Stadiums")
plt.xlabel("Stadium")
plt.ylabel("Number of Matches")
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.show()


# In[31]:


# Sort the DataFrame by stadium counts in ascending order
stadium_counts = df['Stadium'].value_counts().sort_values(ascending=True)
# Plot the count of matches done in each stadium
plt.figure(figsize=(20, 18))
sns.barplot(x=stadium_counts.values, y=stadium_counts.index, palette='viridis')
plt.title("Matches Done in Stadiums")
plt.xlabel("Number of Matches")
plt.ylabel("Stadium")
plt.show()


# In[32]:


# lets show by plotting which team does most runs in wankhede stadium:
Mi = df[df.Team_1 == "Mumbai Indians"]['Total_Runs'].sum()
Csk = df[df.Team_2 == "Chennai Super Kings"]['Total_Runs'].sum()
display(Mi,Csk)
bar = plt.bar(["MI","CSK"],[Mi,Csk],color=['c','m'],label=["MI","CSK"])
plt.legend()
plt.title("Total Runs IPL 2023")
plt.bar_label(bar,label_type="center",color="white")


# In[33]:


# Calculate the maximum total runs scored by each team
max_runs_by_team = df.groupby("Team_Batting")[['Total_Runs']].max()

# Sort the DataFrame in ascending order based on the maximum runs
max_runs_by_team_sorted = max_runs_by_team.sort_values(by='Total_Runs', ascending=True)

# Plot the sorted DataFrame
plt.style.use("ggplot")
bar = max_runs_by_team_sorted.plot.barh(figsize=(25, 22),color="brown")
plt.bar_label(bar.containers[0], label_type="center", color="white")
plt.title("Most Runs Scored By Each Team from 2008 to 2023")
plt.xlabel("Total Runs")
plt.ylabel("Teams")
plt.show()


# In[34]:


# Calculate the total number of sixes hit by each player
df_six = df.groupby("Batsman_Names")[['Sixes']].sum()

# Filter players who hit more than 150 sixes
most_six = df_six[df_six.Sixes > 150]

# Sort the DataFrame in ascending order based on the number of sixes
most_six_sorted = most_six.sort_values(by='Sixes', ascending=True)

# Plot the sorted DataFrame
bar = most_six_sorted.plot.barh(figsize=(15, 10),color="orange")
plt.bar_label(bar.containers[0], label_type="center", color="black")
plt.title("Players Who Hit Most Sixes From 2008 - 2023 (Ascending Order)")
plt.xlabel("Number of Sixes")
plt.ylabel("Player Names")
plt.show()


# In[35]:



# Most sixes from 2008 to 2023
df_six = df.groupby("Batsman_Names")[['Sixes']].sum()
most_six = df_six.Sixes.max()
print(f"Most sixes in IPL history:{most_six}")


# In[36]:


# Calculate the total runs scored by each player in 2023
df_2023 = df[df.Year == 2023]
df_runs = df_2023.groupby("Batsman_Names")[["Runs_Scored"]].sum()

# Get the player with the maximum runs scored
max_runs_player = df_runs.idxmax()
max_runs = df_runs.loc[max_runs_player]

print(f"Most runs in IPL 2023: {max_runs_player} with {max_runs} runs\n")

# Filter players who scored more than 500 runs
player_names = df_runs[df_runs.Runs_Scored > 500]

# Sort the DataFrame in descending order based on runs scored
player_names_sorted = player_names.sort_values(by='Runs_Scored', ascending=False)

# Plot the sorted DataFrame
new = player_names_sorted.plot.bar(figsize=(15, 6),color="green", legend=None)
plt.bar_label(new.containers[0], label_type="center", color="white")
plt.title("Orange Cap Holder: Players with Most Runs in IPL 2023")
plt.xlabel("Total Runs")
plt.ylabel("Player Names")
plt.show()


# In[37]:


# Filter the DataFrame for the year 2023 and runs scored greater than or equal to 100
df1_2023 = df[df.Year == 2023]
df_Runs = df1_2023[df1_2023.Runs_Scored >= 100]

# Sort the DataFrame in descending order based on runs scored
df_Runs_sorted = df_Runs.sort_values(by='Runs_Scored', ascending=False)

# Print the player names who scored more than 100 runs
print("Player names who scored more than 100 runs in 2023:")
print(df_Runs_sorted['Batsman_Names'].values)
print()

# Plot the runs scored by each player in descending order
plt.figure(figsize=(15, 6))
b = df_Runs_sorted['Runs_Scored'].plot.bar(width=0.8, color='skyblue')
plt.bar_label(b.containers[0], label_type="center", color="red")
plt.title("Players Who Scored Century in 2023 (Descending Order)")
plt.xlabel("Runs Scored")
plt.ylabel("Player Names")
plt.show()


# In[38]:


# Filter the DataFrame for the year 2023
df2_2023 = df[df.Year == 2023]

# Group by batsman names and get the maximum strike rate
df_str = df2_2023.groupby("Batsman_Names")[["Strike_Rate"]].max()

# Filter for players with a strike rate above 300
df_str = df_str[df_str.Strike_Rate > 300]

# Sort the DataFrame in ascending order based on the strike rate
df_str_sorted = df_str.sort_values(by='Strike_Rate', ascending=True)

# Plot the players with a strike rate above 300 in ascending order
b2 = df_str_sorted.plot.bar(figsize=(15,6),color="blue")
plt.bar_label(b2.containers[0], label_type="center", color="black")
plt.title("Players Who Had Strike Rate Above 300 in 2023 (Ascending Order)")
plt.show()


# In[39]:


# lets see the numerical and categorical columns:
cat_col = df.select_dtypes(include = object)
num_col = df.select_dtypes([int,float])
display(cat_col.columns,"\n",num_col.columns,"\n\n")

# we can fetch both categorical and numerical columns by pyhton approach
ca_col = [i for i in df.columns if df[i].dtypes == "O"]

nm_col = [i for i in df.columns if i not in ca_col]
print(ca_col,"\n",nm_col)


# In[40]:


def hist_box_plots(data,col,bins="auto"):
    fig,axis = plt.subplots(ncols=2,figsize=(11,3)) 
    sns.histplot(data=data,x=col,bins=bins,ax=axis[0],kde=True)
    sns.boxplot(data=data,x=col,ax=axis[1])


# In[41]:


for col in num_col.columns:
    hist_box_plots(num_col,col)


# In[42]:


# which Team performance very low in 2023:
plt.style.use("classic")
df_2023 = df[df.Year == 2023]

df_per = df_2023.groupby("Team_Batting")[["Total_Runs"]].min()
df_per = df_per[df_per.Total_Runs <= 120]

b = df_per.plot.barh(figsize=(15,6),color="skyblue")
plt.bar_label(b.containers[0],label_type="center",color="brown")
plt.title("Less Runs Scored by Teams in 2023")


# # # machine learning
# 

# In[43]:


from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import sklearn.metrics as mt
from sklearn.model_selection import train_test_split 


# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns
numeric_df = df.select_dtypes(include=['number'])
plt.figure(figsize=(14,10))
sns.heatmap(numeric_df.corr(), annot=True)
plt.show()


# In[45]:


numeric_df.corr()


# In[46]:


# dependent and independent variables
x=df[["IPL_Edition","Total_Runs","Balls_Played","Runs_Scored","Fours","Sixes","Strike_Rate","Year"]]
y=df["Strike_Rate"]


# In[47]:



# Dividing the data into train and test
xtrain,  xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2,random_state=42)


# In[48]:



print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)


# In[49]:


#linear_regression
m = LinearRegression()
m.fit(xtrain,ytrain)


# In[50]:



# Calculate the accuracy score
print("Linear Regressor accuracy score:",m.score(xtest,ytest))


# In[51]:



output= m.predict(xtest)
output


# In[52]:



ytest


# In[53]:



compaire = pd.DataFrame({'actual_value':ytest,'predict_value':output})
compaire


# In[54]:



from sklearn import metrics
mean_aberror = metrics.mean_absolute_error(ytest,output)
mean_sqerror = metrics.mean_squared_error(ytest,output)
rmsqurrerror = np.sqrt(metrics.mean_squared_error(ytest,output))
print(m.score(x,y)*100)
print(mean_aberror) 
print(mean_sqerror)
print(rmsqurrerror) 


# In[55]:



#k-nearest neighbors
from sklearn.neighbors import KNeighborsRegressor
knn_model=KNeighborsRegressor(n_neighbors=3)
knn_model.fit(xtrain,ytrain)


# In[56]:



# Calculate the accuracy score
print("Knn_Model accuracy score:", knn_model.score(xtrain,ytrain))


# In[57]:


#decision tree
dt_model = DecisionTreeRegressor()
dt_model.fit(xtrain, ytrain)


# In[58]:



# Calculate the accuracy score
print("Decision tree accuracy score:", dt_model.score(xtrain, ytrain))


# In[59]:


#random forest
rf_model = RandomForestRegressor()
rf_model.fit(xtrain, ytrain)


# In[60]:



# Calculate the accuracy score
print("RandomForest Regressor accuracy score:", rf_model.score(xtrain, ytrain))


# In[61]:



#Y = β0 + β1X1 + β2X2 + β3X3 + … + βnXn + e
print("intercept i.e b0",m.intercept_)
print("coefficients i.e b1,b2,b3")
list(zip(x,m.coef_))




# In[68]:


from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[69]:


# Train KNN model
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(xtrain, ytrain)
knn_mse = mean_squared_error(ytrain, knn_model.predict(xtrain))

# Train Decision Tree model
dt_model = DecisionTreeRegressor()
dt_model.fit(xtrain, ytrain)
d_mse = mean_squared_error(ytrain, dt_model.predict(xtrain))

# Train Random Forest model
rf_model = RandomForestRegressor()
rf_model.fit(xtrain, ytrain)
r_mse = mean_squared_error(ytrain, rf_model.predict(xtrain))

# Plotting
models = ['KNN', 'Decision Tree', 'Random Forest']
mse_values = [knn_mse, d_mse, r_mse]

plt.barh(models, mse_values, color=['red', 'green', 'blue'])
plt.xlabel('Models')
plt.ylabel('Mean Squared Error')
plt.title('Mean Squared Error Comparison for Different Models')
plt.show()


# In[ ]:




