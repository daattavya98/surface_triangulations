# Surface Triangulations

A Python pipeline for generating random triangulations of closed 2-dimensional surfaces. The library produces combinatorial triangulations (represented as incidence matrices) for:

- **Spheres** (genus 0)
- **Tori** (genus 1) and higher-genus orientable surfaces
- **Klein bottles** and other non-orientable surfaces
- **Disjoint unions** of the above (e.g. sphere + torus)

Each triangulation is constructed by sampling random points on a fundamental polygon, computing a Delaunay triangulation, and then performing the appropriate edge identifications to obtain a valid simplicial complex. The pipeline also verifies that the resulting complex is a genuine surface by checking Betti numbers and the manifold link condition.

---

## Installation

The project uses [Poetry](https://python-poetry.org/) for dependency management. You will need Python **3.10 or 3.11**.

### 1. Install Poetry

If you don't have Poetry installed, the recommended way is via `pipx`:

```bash
pipx install poetry
```

Alternatively, on macOS you can use Homebrew:

```bash
brew install poetry
```

### 2. Clone the repository

```bash
git clone https://github.com/daattavya98/ai_lakatos.git
cd ai_lakatos
```

### 3. Install dependencies

```bash
poetry install
```

This creates a virtual environment and installs all required packages.

### 4. Activate the environment

Either prefix commands with `poetry run`, or activate the shell:

```bash
poetry shell
```

---

## Quick Start: Generating Sphere Triangulations

The main entry point for generating triangulations is `surface_triangulations.data_gen.end_to_end_gen`.

### Generate a single sphere triangulation

```python
from surface_triangulations.data_gen.end_to_end_gen import generate_genus_0_datapoints

# Returns the vertex-edge (D1) and edge-face (D2) incidence matrices
D1, D2 = generate_genus_0_datapoints()

print("Vertex-edge incidence matrix shape:", D1.shape)
print("Edge-face incidence matrix shape:", D2.shape)
```

The function internally:
1. Samples random points on two fundamental squares.
2. Computes a Delaunay triangulation on each square.
3. Identifies boundary edges to form a sphere.
4. Verifies the result has the correct Betti numbers (b0 = 1, b1 = 0, b2 = 1).

You can control the complexity of the triangulation by passing parameters:

```python
D1, D2 = generate_genus_0_datapoints(
    n_cycle_1=4,            # points on the first identified edge pair
    n_cycle_2=5,            # points on the second identified edge pair
    n_cycle_3=4,            # points on the third identified edge pair
    n_cycle_4=6,            # points on the fourth identified edge pair
    n_interior=5,           # interior points in the first square
    n_interior_square_2=5,  # interior points in the second square
    n_diagonal_1_square_1=3,
    n_diagonal_2_square_1=3,
    n_diagonal_1_square_2=3,
    n_diagonal_2_square_2=3,
)
```

### Generate a dataset of sphere triangulations

To generate a batch of unique sphere triangulations and save them:

```python
from surface_triangulations.data_gen.end_to_end_gen import generate_genus_0_dataset

# Generate 50 sphere triangulations with random parameters
dataset = generate_genus_0_dataset(
    n_lower=5,         # minimum number of points per parameter
    n_upper=25,        # maximum number of points per parameter
    no_of_points=50,   # total number of triangulations to generate
)

# dataset is a (50, 2) numpy array of (D1, D2) pairs
print("Dataset shape:", dataset.shape)
```

### Other surfaces

```python
from surface_triangulations.data_gen.end_to_end_gen import (
    generate_genus_1_datapoint,     # single torus
    generate_genus_1_dataset,       # batch of tori
    generate_genus_2_datapoint,     # genus-2 surface
    generate_klein_bottle_datapoints,  # Klein bottles
    generate_disconnected_datapoint,   # disjoint union of orientable surfaces
    generate_arbitrary_disjoint_union_datapoint,  # arbitrary disjoint unions
)
```

---

## Project Structure

```
surface_triangulations/
├── data_gen/
│   ├── end_to_end_gen.py        # Main generation pipeline
│   ├── constructing_sc.py       # Simplicial complex construction utilities
│   ├── top_mfld_check.py        # Topological manifold verification
│   ├── data_visualisation.py    # Plotting utilities
│   └── incidence_matrix_dataframes/  # Pre-generated CSV datasets
├── data_preprocessing.py        # Dataset validation and feature extraction
└── tests/
```

---

## License

MIT
