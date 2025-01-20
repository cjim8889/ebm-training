import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize wandb
api = wandb.Api()

def get_std_dt_logZt(runid="8gnjw966"):
    run_path = f"jzinou/liouville/{runid}"
    run = api.run(run_path)
    history = run.scan_history(
        keys=[
            "_step", 
            "average_loss"
        ],  # Specify the metric you want
    )

    steps = []
    average_loss = []
    for row in history:
        if row["_step"] > 50*1000:
            break
        steps.append(row["_step"])
        average_loss.append(row["average_loss"])

    return steps, average_loss

steps, loss_mcmc = get_std_dt_logZt(runid="8gnjw966")
_, loss_mcmc_velocity = get_std_dt_logZt(runid="927tu038")

# add ema smoothing
df = pd.DataFrame({'loss_mcmc': loss_mcmc,})
loss_mcmc = df['loss_mcmc'].ewm(span=5).mean()
df = pd.DataFrame({'loss_mcmc_velocity': loss_mcmc_velocity,})
loss_mcmc_velocity = df['loss_mcmc_velocity'].ewm(span=5).mean()

plt.figure(figsize=(12, 8))
plt.grid(True, which="both", ls="--", linewidth=.5)

plt.plot(steps, np.log(loss_mcmc), 'r-', label=r'$\mathbb{E}_{p_t} \partial_t \log \tilde{p}_t (x)$', linewidth=2)
plt.plot(steps, np.log(loss_mcmc_velocity), 'b-', label=r'$\mathbb{E}_{p_t} \xi_t (x;v_t)$', linewidth=2)
plt.xticks(1000*np.array([0, 10, 20, 30, 40, 50]), ['0', '10K', '20K', '30K', '40K', '50K'])
# plt.ylim(0, 1.0)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=30)
plt.xlabel('Training Steps', fontsize=28)
plt.ylabel(r'$\log$-Loss', fontsize=28)
plt.tight_layout()

# Add spines and ticks styling
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.savefig("loss_plot.png", dpi=300, bbox_inches='tight', pad_inches=0.01)
