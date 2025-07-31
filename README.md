# bubble_bouncing

Simulation code for the bubble bouncing project. The basic model uses the thin film equation and a force balance model to simulate the shape and motion of a bubble when bouncing into a solid surface (as in Manica 2015, Esmaili 2019 and Hooshanginejad 2023). 

On top of existing works, the current simulation code features:

1. Oseen wake induced circulation, which leads to a lift force;
2. Object oriented programming with improved readability and maintainability;
3. Data saveing with high performance .h5 files;
4. Visualization tools. 

## Note for future development

Import within the package is adapted to the "editable install" style. That said, internal imports are only valid once this package is installed with

```
pip install -e .
```

A fresh and minimal environment is recommended, and can be set up using `conda`

```
conda env create -f environment.yaml
conda activate bcsim
```

If new dependencies are added in the future, update `environment.yaml` using

```
conda env export --from-history > environment.yaml
```

## Versions

- 0.1.0: Bubble bouncing with Oseen wake circulation induced lift.
- 0.1.1: Implement BCEP001.
- 0.1.2: Fix bug. Include lift force in the force calculation.
- 0.1.3: Fix bug. (i) Change the sign of unit tangent vectors on bubble surface to make it consistent with the lift force calculation. (ii) Fix the Oseen wake calculation for arbitrary bubble velocity.
- 0.2.0: Implement BCEP005. BCEP002 and BCEP003 are also implemented, but moved to a separate repo for better maintainance logic.

## Enhancement proposals (BCEP)

#### BCEP001: improve visualization methods (v0.1.1)

1. Circulation visualization method;
2. Bubble deformation visualization method;
3. Animation generator: all vis scripts should be able to generate preview (one image with selected frames drawn), video (480p .mp4, use flag -v as "video") and HD video (1080p .mp4, use flag -vh as "high quality video").

#### BCEP002: improve camera motion and playback speed (v0.2.0)

1. Camera should follow the bubble at the point of bouncing;
2. Playback can slow down to better visualize the bounce;
3. Transparent surface can help.

#### BCEP003: Make visualize.py more modular (v0.2.0)

The current form is difficult to maintain, due to the various mode options and the combined dynamic and static objects. Ideally, this class should be like the manim main class, where adding a reference box should just be a one-liner. There should be methods such as `add_bubble`, `add_surface`, `camera_closeup(object)`. Then, the completed scene would control the export options, such as `w`, `s`, `v` and `vh`. 

#### BCEP004: Resume simulation

To load parameters and current state from file and resume the simulation.

#### BCEP005: Separate simulation and visualization code (v0.2.0)

The function of one package should be more focused. Also the visualization code is more needed in the analysis routines. 