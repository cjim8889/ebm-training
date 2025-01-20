import wandb
import numpy as np
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
            "mean_std/std_dt_log_Zt_mcmc",
            "mean_std/std_dt_log_Zt_mcmc_velocity",
        ],  # Specify the metric you want
    )

    steps = []
    std_dt_log_Zt_mcmc = []
    std_dt_log_Zt_mcmc_velocity = []
    for row in history:
        if row["_step"] > 50*1000:
            break
        steps.append(row["_step"])
        std_dt_log_Zt_mcmc.append(row["mean_std/std_dt_log_Zt_mcmc"])
        std_dt_log_Zt_mcmc_velocity.append(row["mean_std/std_dt_log_Zt_mcmc_velocity"])

    return steps, std_dt_log_Zt_mcmc, std_dt_log_Zt_mcmc_velocity

steps, std_dt_log_Zt_mcmc, _ = get_std_dt_logZt(runid="8gnjw966")
_, _, std_dt_log_Zt_mcmc_velocity = get_std_dt_logZt(runid="927tu038")
print(len(steps), len(std_dt_log_Zt_mcmc), len(std_dt_log_Zt_mcmc_velocity))


# Create the figure
plt.figure(figsize=(12, 8))
# plt.gca().set_facecolor('#F0F0F0')
plt.grid(True, which="both", ls="--", linewidth=.5)

# Plot lines with confidence intervals
plt.plot(steps, std_dt_log_Zt_mcmc, 'r-', label=r'$\mathbb{E}_{p_t} \partial_t \log \tilde{p}_t (x)$', linewidth=2)
plt.plot(steps, std_dt_log_Zt_mcmc_velocity, 'b-', label=r'$\mathbb{E}_{p_t} \xi_t (x;v_t)$', linewidth=2)
plt.xticks(1000*np.array([0, 10, 20, 30, 40, 50]), ['0', '10K', '20K', '30K', '40K', '50K'])
plt.ylim(0, 1.0)

plt.xticks(fontsize=24)
plt.yticks(fontsize=24)
plt.legend(fontsize=30)
plt.xlabel('Training Steps', fontsize=28)
plt.ylabel(r'Std of $\partial_t \log Z_t$ Estimation', fontsize=28)
plt.tight_layout()

# Add spines and ticks styling
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# ax.spines['left'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['left'].set_color('gray')
# ax.spines['bottom'].set_color('gray')
# ax.tick_params(axis='both', colors='gray')

plt.savefig("std_dt_logZt_plot.png", dpi=300, bbox_inches='tight', pad_inches=0.01)

# history = run.scan_history(
#     keys=[
#         "_step", 
#         "average_loss"
#     ],  # Specify the metric you want
# )
# for row in history:
#     print(row)
#     exit()