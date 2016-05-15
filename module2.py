### Learning Activity 1: Load the required libraries

import scipy
import numpy as np
import pandas as pd
import plotly.plotly as py

import visplots

from plotly.graph_objs import *
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
from sklearn import preprocessing, metrics
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats.distributions import randint

init_notebook_mode()


### Learning Activity 2: Importing the data

# Import the data and explore the first few rows
wineQ  = pd.read_csv("data/wine_quality.csv", sep=",")
header = wineQ.columns.values
wineQ.head()

# Convert to numpy array and check the dimensionality
npArray = np.array(wineQ)
print(npArray.shape)


### Learning Activity 3: Inspect your data by indexing and index slicing

# Print the 1st row and 1st column of npArray
print (npArray[0,0])

# Print the 1st row and 2nd column of npArray
print (npArray[0,1])

# Print the first 3 rows of npArray
print (npArray[:3,])

# Print the first 3 rows from the last column of npArray
print (npArray[:3,10])
# print (npArray[:3,-1])


### Learning Activity 4: Split the data into input features, X, and outputs, y

# Split to input matrix X and class vector y
X = npArray[:,:-1].astype(float)
y = npArray[:,-1]

# Print the dimensions of X and y
print ("X dimensions:", X.shape)
print ("y dimensions:", y.shape)


### Learning Activity 5:  Investigate the y frequencies

# Print the y frequencies
yFreq = scipy.stats.itemfreq(y)
print(yFreq)

# Convert the categorical to numeric values, and print the y frequencies
le = preprocessing.LabelEncoder()
y  = le.fit_transform(y)

yFreq = scipy.stats.itemfreq(y)
print(yFreq)


# Display the y frequencies in a barplot with Plotly
data = [
    Bar(
        x = ['High Quality', 'Low Quality'],
        y = [yFreq[0][1], yFreq[1][1]],
        marker = dict(color=['blue','orange'])
    )
]

layout = Layout(
    xaxis = dict(title = "Wine Quality"),
    yaxis = dict(title = "Count"),
    width = 500,
)

fig = dict(data = data, layout = layout)

iplot(fig)


### Learning Activity 6: Data scaling

wineQ.describe()


# Create a boxplot of the raw data
nrow, ncol = X.shape

data = [
    Box(
        y = X[:,i],        # values to be used for box plot
        name = header[i],  # label (on hover and x-axis)
        marker = dict(color = "purple"),
    ) for i in range(ncol)
]

layout = Layout(
    xaxis = dict(title = "Feature"),
    yaxis = dict(title = "Value"),
    showlegend=False,
)

fig = dict(data = data, layout = layout)

iplot(fig)



# Auto-scale the data
X = preprocessing.scale(X)


# Create a boxplot of the scaled data
data = [
    Box(
        y = X[:,i],
        name = header[i],
        boxpoints='all',
        jitter=0.4,
        whiskerwidth=0.2,
        marker=dict(
            size=2,
        ),
        line=dict(width=1),
        boxmean='sd'
    ) for i in range(X.shape[1])
]

layout = Layout(
    xaxis = dict(title = "Feature", tickangle=40),
    yaxis = dict(title = "Value"),
    showlegend=False,
    height=700,
    margin=Margin(b=170, t=50),
)

fig = dict(data = data, layout = layout)

iplot(fig)


### Learning Activity 7: Investigate the relationship between input features

# Create an enhanced scatter plot of the first two features
f1 = 0
f2 = 1

# Low quality (class "1") represented with orange x
trace1 = Scatter(
    x = X[y == 1, f1],
    y = X[y == 1, f2],
    mode = 'markers',
    name = 'Low Quality ("1")',
    marker = dict(
        color  = 'orange',
        symbol = 'x'
    )
)

# High quality (class "0") represented with blue circles
trace2 = Scatter(
    x = X[y == 0, f1],
    y = X[y == 0, f2],
    mode = 'markers',
    name = 'High Quality ("0")',
    marker = dict(
        color  = 'blue',
        symbol = 'circle'
    )
)

layout = Layout(
    xaxis = dict(title = header[f1]),
    yaxis = dict(title = header[f2]),
    height= 600,
)

fig = dict(data = [trace1, trace2], layout = layout)

iplot(fig)


### Bonus Activity 8: Try plotting different combinations of three features
### (f1, f2, f3) in the same plot.


# Create a 3D scatterplot using the first three features
f1 = 0
f2 = 1
f3 = 2

desc = dict(
    classes = [1, 0],
    colors  = ["orange", "blue"],
    labels  = ['Low Quality ("1")', 'High Quality ("0")'],
    symbols = ["x", "circle"]
)

data = [
    Scatter3d(
        x = X[y == desc["classes"][i], f1],
        y = X[y == desc["classes"][i], f2],
        z = X[y == desc["classes"][i], f3],
        name = desc["labels"][i],
        mode = "markers",
        marker = dict(
            size = 2.5,
            symbol = desc["symbols"][i],
            color  = desc["colors"][i]
        )
    ) for i in range(len(desc["labels"]))
]

layout = Layout(
    scene=Scene(
        xaxis=XAxis(title=header[f1], titlefont=dict(size=11)),
        yaxis=YAxis(title=header[f2], titlefont=dict(size=11)),
        zaxis=ZAxis(title=header[f3], titlefont=dict(size=11))
    ),
    margin=Margin(l=80, r=80, b=0, t=0, pad=0, autoexpand=True),
    height= 600,
)

fig = dict(data = data, layout = layout)

iplot(fig)


### Bonus Activity 9: Try different combinations of f1 and f2 (in a
### grid/scatterplot matrix if you can).

# Create a grid plot of scatterplots using a combination of features

from plotly import tools

fig = tools.make_subplots(rows=4, cols=4, shared_xaxes=True, shared_yaxes=True)

for row in range(0, 4):
    for col in range(0, 4):
        # orange x, Low quality
        trace1 = Scatter(
            x = X[y == 1, col],
            y = X[y == 1, row],
            mode = 'markers',
            marker = dict(
                color  = 'orange',
                symbol = 'x',
                opacity = .5
            )
        )
        # blue circles, High quality
        trace2 = Scatter(
            x = X[y == 0, col],
            y = X[y == 0, row],
            mode = 'markers',
            marker = dict(
                color  = 'blue',
                symbol = 'circle',
                opacity = .5
            )
        )
        posX = row+1
        posY = col+1
        fig.append_trace(trace1, posX, posY)
        fig.append_trace(trace2, posX, posY)
        fig['layout']['xaxis'+str(posX)].update(title=header[row])
        fig['layout']['yaxis'+str(posY)].update(title=header[col])

fig['layout'].update(
    showlegend=False,
    height=900,
)

iplot(fig)



### Bonus Activity 10: Create a correlation matrix and plot a heatmap of
### correlations between the input features in X

# Calculate the correlation coefficient
correlationMatrix = np.corrcoef(X, rowvar=0)

# Create a heatmap of the correlation coefficients
data = [
    Heatmap(
        x = header,             # sites on both
        y = header,             #  axes
        z = correlationMatrix,  # correlation as color contours
        colorscale='RdOrBl',    # light yellow-orange-red colormap
        reversescale=True       # inverse colormap order
    )
]

layout = Layout(
    xaxis = dict(title = "Feature"),
    yaxis = dict(title = "Feature"),
    margin= Margin(l=250),
    height = 700,
)

fig = dict(data = data, layout = layout)

iplot(fig)