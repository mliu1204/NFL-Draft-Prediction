import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from svgpath2mpl import parse_path

def convertToPoints(svg_path, duration, debug=False):
    path = parse_path(svg_path)
    coordinates = np.array(path.to_polygons(closed_only=False))
    coordinates[0][:,1] = 100.0-coordinates[0][:,1]
    data = coordinates.T[:,:,0].tolist()
    x, y = data[0]/np.max(data[0])*duration, data[1]
    f = interp1d(x, y)
    xnew = np.arange(0, duration)
    ynew = f(xnew)
    return ynew.tolist()

df = pd.read_csv('trimmed_youtube.csv')
print(f"ID: {df.loc[0, 'id']}")
print(f"View Count: {df.loc[0, 'viewCount']}")
print(f"Date: {df.loc[0, 'date']}")


# Sort the dataframe by viewCount in descending order
# df = df.sort_values(by='viewCount', ascending=False)

# print(df.loc[0, 'retentionCurve'])
# retention_points = convertToPoints(df.loc[0, 'retentionCurve'], df.loc[0, 'duration'])

# # Create the plot
# plt.figure(figsize=(10, 6))
# plt.plot(retention_points, linewidth=2)
# plt.title('YouTube Video Retention Curve')
# plt.xlabel('Time (seconds)')
# plt.ylabel('Retention (%)')
# plt.grid(True)
# plt.xlim(left=0)
# plt.show()