# CasADi Optimization Examples 🚀

A small collection of **CasADi + IPOPT** demos that solve classic
non‑linear programming and optimal‑control problems, then visualize the results.

| Script | Problem | Method |
|--------|---------|--------|
| `NLP_example.py` | Toy nonlinear program (Rosenbrock‑style) | IPOPT |
| `cartpole_swingup_ipopt.py` | Cart‑pole swing‑up OCP | Direct multiple‑shooting + IPOPT |
| `damped_double_pendulum_ipopt.py` | Damped double‑pendulum swing | Direct multiple‑shooting + IPOPT |

---

## 📦 Quick start

```bash
conda create -n casadi python=3.10
conda activate casadi
pip install casadi matplotlib numpy

# run a demo (generates plots & GIF)
python cartpole_swingup_ipopt.py
```

---

## 📊 Sample Outputs


<p align="center"><b>Cart‑Pole Swingup</b></p>
<p align="center">
  <img src="plots and animations/cartpole_swingup.png" width="50%"/>
</p>


<p align="center"><b>Damped Double Pendulum Swingup</b></p>

  <p align="center">
    <img src="plots and animations/Damped_double_pendulum_swingup.png" width="50%"/>
  </p>
  <p align="center">
    <img src="plots and animations/Damped_double_pendulum_trajectory.gif" width="50%"/>
  </p>
</details>
