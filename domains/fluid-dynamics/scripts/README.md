## 🔧 Run Commands

```bash
python3 ccns_c_reference_clean.py \
  --Nx 200 --Ny 128 --Tfinal 0.08 --CFL 0.2 \
  --Re 5000 --kappa 1.0 --theta_mu 1.0
```

```bash
python3 ccns_c_reference_clean.py \
  --Nx 200 --Ny 128 --Tfinal 1.0 --CFL 0.2 --Re 20000 \
  --kappa 1.0 --theta_mu 1.0
```

```bash
python3 ccns_c_reference_clean.py --kappa 0.3
python3 ccns_c_reference_clean.py --kappa 0.4
python3 ccns_c_reference_clean.py --kappa 0.6 
