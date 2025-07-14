import numpy as np
import pyvista as pv
import time

# Create data
x = np.linspace(0, 4 * np.pi, 100)
y = np.sin(x)
points = np.c_[x, y, np.zeros_like(x)]

# Create PolyData and plotter
cloud = pv.PolyData(points)
pl = pv.Plotter()
pl.add_mesh(cloud, point_size=10, render_points_as_spheres=True)
pl.show(interactive=True, interactive_update=True)

# Live update loop
for i in range(100):
    # Update y values
    y = np.sin(x + i * 0.1)
    new_points = np.c_[x, y, np.zeros_like(x)]

    # Update mesh points
    cloud.points = new_points

    pl.update()
    # time.sleep(0.05)  # simulate real-time update
