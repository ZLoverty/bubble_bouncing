# bubble_bouncing

Simulation code for the bubble bouncing project. The basic model uses the thin film equation and a force balance model to simulate the shape and motion of a bubble when bouncing into a solid surface (as in Manica 2015, Esmaili 2019 and Hooshanginejad 2023). 

On top of existing works, the current simulation code features:

1. Oseen wake induced circulation, which leads to a lift force;
2. Object oriented programming with improved readability and maintainability;
3. Data saveing with high performance .h5 files;
4. Visualization tools. 

## Enhancement proposals (BCEP)

### BCEP001: improve visualization methods

1. Circulation visualization method;
2. Bubble deformation visualization method;
3. Animation generator: all vis scripts should be able to generate preview (one image with selected frames drawn), video (480p .mp4, use flag -v as "video") and HD video (1080p .mp4, use flag -vh as "high quality video").

