# UAV Strategy Engine — Genetic Algorithm (AI Assignment 2, CLO4)

Multi-objective GA that evolves deployment positions, waypoint trajectories, speeds and
sensor-power profiles for a fleet of 3 UAVs over a 50x50 hazard grid.

## Project structure
```
uav_ga_project/
├── main.py                  # run GA + random-search baseline + all plots
├── experiments.py           # hyper-parameter sensitivity analysis
├── uniqueness.py            # 5-run GA vs random-search uniqueness study
├── requirements.txt
├── report.md / report.docx  # technical report
├── uav_ga/
│   ├── config.py            # every parameter (grid, weights, GA settings)
│   ├── environment.py       # grid: no-fly zones, hazards, priority zones, targets
│   ├── chromosome.py        # chromosome encoding (UAV x waypoint x [x,y,speed,power])
│   ├── fitness.py           # multi-objective fitness + dynamic penalty
│   ├── constraints.py       # repair mechanisms
│   ├── operators.py         # tournament selection, crossover, mutation
│   ├── ga.py                # GeneticAlgorithm class, random-search baseline
│   └── visualize.py         # all figures
└── outputs/                 # generated plots, results.json, sensitivity.csv
```

## How to run (VS Code terminal)
```bash
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python main.py                 # ~20 s  -> outputs/fig1..fig4, results.json
python experiments.py          # ~3-4 min -> outputs/fig5_sensitivity.png, sensitivity.csv
python uniqueness.py           # ~2 min   -> outputs/fig6_uniqueness.png, uniqueness.csv
```

## Outputs
| File | Content |
|---|---|
| fig1_convergence.png | Best vs average fitness (+ random search), diversity, mutation rate |
| fig2_final_paths_heatmap.png | Final evolved paths + coverage heat-map |
| fig3_generations.png | Paths & heat-maps at generations 0, 37, 75, 150 |
| fig4_ga_vs_random.png | GA solution vs random-search solution |
| fig5_sensitivity.png | Population size / mutation rate / selection pressure study |
| fig6_uniqueness.png | Final solutions of 5 independent runs: GA vs random search |
