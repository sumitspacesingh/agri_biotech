import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# ==================================================================
# DASHBOARD PAGE SETUP
# ==================================================================
st.set_page_config(page_title="Agri-Biotech Risk Dashboard", layout="wide")

st.title("🌾 Agricultural Biotech Supply Chain Risk Model")
st.caption(
    "Multi-factor infection risk engine with regional contagion correlation, insurance, "
    "and Monte Carlo tail-risk simulation. Educational/illustrative model — calibrate on "
    "real agronomic and claims data before operational use."
)
st.markdown("---")

# ==================================================================
# SIDEBAR — INPUTS
# ==================================================================
st.sidebar.header("🚜 Portfolio Structure")
num_farms = st.sidebar.number_input("Number of Contracted Farms", min_value=1, value=100, step=10)
asset_value_mean = st.sidebar.number_input("Mean Asset Value per Farm (₹)", min_value=1000, value=1000000, step=50000)
asset_value_cv = st.sidebar.slider("Asset Value Variability (Coefficient of Variation, %)", 0, 100, 25) / 100.0
num_regions = st.sidebar.slider(
    "Number of Distinct Growing Regions", 1, 20, 4,
    help="More regions with independent weather/disease exposure = more diversification benefit."
)

st.sidebar.header("🔬 Biological Risk")
p_baseline = st.sidebar.slider("Baseline Infection Probability (%)", 0.0, 100.0, 15.0) / 100.0
rho_infection = st.sidebar.slider(
    "Within-Region Contagion Correlation (ρ)", 0.0, 0.95, 0.30,
    help="How strongly infections at nearby farms move together (spatial/vector-borne spread)."
)
lgi_base = st.sidebar.slider("Loss Given Infection / LGI (%)", 0.0, 100.0, 80.0) / 100.0
detection_speed = st.sidebar.slider(
    "Early-Detection / Quarantine Effectiveness (%)", 0, 100, 20,
    help="Faster detection and quarantine reduces the share of crop value actually lost once infected."
) / 100.0

st.sidebar.header("🛡️ Mitigation Strategy")
shield_adoption_pct = st.sidebar.slider("Bio-Shield Adoption Rate (% of farms)", 0, 100, 100) / 100.0
shield_efficacy_pct = st.sidebar.slider("Bio-Shield Efficacy (% reduction in infection probability)", 0, 100, 80) / 100.0
shield_cost_per_farm = st.sidebar.number_input("Bio-Shield Cost per Adopting Farm (₹)", min_value=0, value=20000, step=1000)

st.sidebar.header("📄 Insurance / Risk Transfer")
insurance_coverage_pct = st.sidebar.slider("Indemnity Coverage (% of loss reimbursed)", 0, 100, 0) / 100.0
insurance_premium_per_farm = st.sidebar.number_input("Insurance Premium per Farm (₹)", min_value=0, value=0, step=500)

st.sidebar.header("⚡ Stress Testing")
p_stress_multiplier = st.sidebar.slider("Climate Anomaly: Infection Probability Multiplier", 1.0, 5.0, 2.5, 0.1)
rho_stress_add = st.sidebar.slider(
    "Climate Anomaly: Additional Contagion Correlation", 0.0, 0.5, 0.20,
    help="Outbreaks tend to synchronize across farms more than normal-year infections."
)

st.sidebar.header("🎲 Simulation Settings")
num_simulations = st.sidebar.slider("Monte Carlo Trials", 500, 5000, 2000, 500)
var_confidence = st.sidebar.selectbox("Tail Risk Confidence Level", [95, 99], index=1)

# ==================================================================
# INPUT VALIDATION
# ==================================================================
errors = []
if num_farms < num_regions:
    errors.append("Number of farms must be at least the number of regions.")
if p_baseline <= 0:
    errors.append("Baseline infection probability must be greater than zero for the model to be meaningful.")
if errors:
    for e in errors:
        st.error(e)
    st.stop()

# ==================================================================
# PORTFOLIO CONSTRUCTION (fixed across simulations for stability)
# ==================================================================
rng = np.random.default_rng(42)

# Assign farms round-robin to regions (independent weather/disease exposure per region)
region_id = np.arange(num_farms) % num_regions

# Farm asset values drawn from a lognormal distribution around the mean, with given variability
if asset_value_cv > 0:
    sigma = np.sqrt(np.log(1 + asset_value_cv ** 2))
    mu = np.log(asset_value_mean) - sigma ** 2 / 2
    farm_asset_values = rng.lognormal(mean=mu, sigma=sigma, size=num_farms)
else:
    farm_asset_values = np.full(num_farms, float(asset_value_mean))

# Which farms adopted the bio-shield (random subset matching adoption rate)
adopted = rng.random(num_farms) < shield_adoption_pct
num_adopters = int(adopted.sum())

# Effective LGI after early-detection/quarantine effectiveness
lgi_effective = lgi_base * (1 - detection_speed)

# ==================================================================
# CORRELATED MONTE CARLO ENGINE (single-factor-per-region contagion model)
# ==================================================================
def simulate_losses(
    p_farm: np.ndarray,
    rho: float,
    n_sims: int,
    region_id_arr: np.ndarray = None,
    n_regions: int = None,
) -> np.ndarray:
    """
    Simulate portfolio losses under a Gaussian single-factor model per region.
    Farms in the same region share a systemic shock Z_region; independent regions
    give a diversification benefit. Farm i is 'infected' in a trial if its latent
    variable falls below norm.ppf(p_farm[i]).

    region_id_arr / n_regions default to the portfolio's actual region assignment,
    but can be overridden (e.g. for the diversification sensitivity sweep below)
    without touching global state.
    """
    if region_id_arr is None:
        region_id_arr = region_id
    if n_regions is None:
        n_regions = num_regions

    rho_c = np.clip(rho, 0.0, 0.95)
    z_region = rng.standard_normal((n_sims, n_regions))             # regional systemic factor
    eps = rng.standard_normal((n_sims, num_farms))                  # idiosyncratic farm factor
    z_farm = z_region[:, region_id_arr]                             # broadcast region factor to its farms

    latent = np.sqrt(rho_c) * z_farm + np.sqrt(1 - rho_c) * eps
    threshold = norm.ppf(np.clip(p_farm, 1e-6, 1 - 1e-6))
    infected = latent < threshold                                   # shape (n_sims, num_farms)

    loss_per_farm_if_infected = farm_asset_values * lgi_effective * (1 - insurance_coverage_pct)
    portfolio_losses = infected @ loss_per_farm_if_infected          # (n_sims,) vector of trial losses
    return portfolio_losses

def scenario_stats(losses: np.ndarray, fixed_cost: float, confidence: int) -> dict:
    var = np.percentile(losses, confidence)
    tail = losses[losses >= var]
    cvar = tail.mean() if len(tail) > 0 else var
    return {
        "EL": losses.mean() + fixed_cost,
        "VaR": var + fixed_cost,
        "CVaR": cvar + fixed_cost,
        "UL": var - losses.mean(),  # unexpected loss above expected, before fixed costs
        "losses": losses,
    }

# ---- Scenario 1: Baseline (no shield, no insurance, normal conditions) ----
p_baseline_farms = np.full(num_farms, p_baseline)
losses_baseline = simulate_losses(p_baseline_farms, rho_infection, num_simulations)
fixed_cost_baseline = 0.0
stats_baseline = scenario_stats(losses_baseline, fixed_cost_baseline, var_confidence)

# ---- Scenario 2: Mitigated (shield on adopters + insurance active, normal conditions) ----
p_mitigated_farms = np.where(adopted, p_baseline * (1 - shield_efficacy_pct), p_baseline)
losses_mitigated = simulate_losses(p_mitigated_farms, rho_infection, num_simulations)
fixed_cost_mitigated = num_adopters * shield_cost_per_farm + num_farms * insurance_premium_per_farm
stats_mitigated = scenario_stats(losses_mitigated, fixed_cost_mitigated, var_confidence)

# ---- Scenario 3: Stressed (climate anomaly, no mitigation) ----
p_stressed_farms = np.clip(p_baseline_farms * p_stress_multiplier, 0, 0.99)
rho_stressed = min(rho_infection + rho_stress_add, 0.95)
losses_stressed = simulate_losses(p_stressed_farms, rho_stressed, num_simulations)
stats_stressed = scenario_stats(losses_stressed, 0.0, var_confidence)

# ---- Scenario 4: Stressed WITH mitigation (does the shield hold up under an outbreak?) ----
p_stressed_mitigated_farms = np.where(
    adopted, p_stressed_farms * (1 - shield_efficacy_pct), p_stressed_farms
)
losses_stressed_mitigated = simulate_losses(p_stressed_mitigated_farms, rho_stressed, num_simulations)
stats_stressed_mitigated = scenario_stats(losses_stressed_mitigated, fixed_cost_mitigated, var_confidence)

net_savings = stats_baseline["EL"] - stats_mitigated["EL"]
var_reduction = stats_baseline["VaR"] - stats_mitigated["VaR"]

# ==================================================================
# HEADLINE METRICS
# ==================================================================
st.header("📈 Financial Exposure Analysis")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Baseline Expected Loss", f"₹{stats_baseline['EL']:,.0f}")
with col2:
    st.metric("Mitigated Total Cost (Loss + Shield + Insurance)", f"₹{stats_mitigated['EL']:,.0f}")
with col3:
    if net_savings > 0:
        st.metric("Net Mitigation Savings", f"₹{net_savings:,.0f}", "Recommended on EL basis")
    else:
        st.metric("Net Mitigation Loss", f"-₹{abs(net_savings):,.0f}", "Not recommended on EL basis", delta_color="inverse")
with col4:
    st.metric(f"VaR({var_confidence}%) Reduction from Mitigation", f"₹{var_reduction:,.0f}",
               help="Reduction in tail-risk loss, not just average loss — matters for solvency, not just P&L.")

col5, col6, col7 = st.columns(3)
with col5:
    st.metric(f"Baseline VaR({var_confidence}%)", f"₹{stats_baseline['VaR']:,.0f}")
with col6:
    st.metric(f"Baseline CVaR({var_confidence}%)", f"₹{stats_baseline['CVaR']:,.0f}",
               help="Average loss in the worst (100-confidence)% of outcomes — the true tail exposure.")
with col7:
    st.metric("Contagion-Driven Unexpected Loss (Baseline)", f"₹{stats_baseline['UL']:,.0f}",
               help="Extra loss above the average, driven by correlated/clustered infection risk.")

st.markdown("---")

# ==================================================================
# DECISION ENGINE
# ==================================================================
decision_col, notes_col = st.columns(2)

with decision_col:
    st.subheader("✅ Mitigation Decision")
    if net_savings > 0 and var_reduction > 0:
        st.success(
            f"**DEPLOY THE BIO-SHIELD** — it reduces both average loss (by ₹{net_savings:,.0f}) "
            f"and tail risk (VaR by ₹{var_reduction:,.0f}). Clear case on both P&L and solvency grounds."
        )
    elif net_savings > 0 and var_reduction <= 0:
        st.warning(
            "**MARGINAL CASE** — the shield reduces average expected loss but does not meaningfully "
            "reduce tail risk (perhaps because contagion correlation dominates outcomes even with lower "
            "individual-farm probabilities). Consider combining with insurance or regional diversification."
        )
    else:
        st.error(
            f"**DO NOT DEPLOY AS CONFIGURED** — upfront shield/insurance cost "
            f"(₹{fixed_cost_mitigated:,.0f}) outweighs the reduction in expected loss "
            f"(-₹{abs(net_savings):,.0f}). Revisit adoption rate, efficacy assumptions, or cost."
        )
    st.caption(
        "Decision weighs both expected loss (P&L view) and VaR (tail-risk view) — a mitigation that "
        "only helps on average but not in bad scenarios is a weaker case than one that helps on both."
    )

with notes_col:
    st.subheader("💡 Risk Committee Notes")
    st.info(
        f"**Climate Anomaly Stress (no shield):** infection probability rises to "
        f"**{p_stress_multiplier:.1f}×** baseline and contagion correlation rises to **{rho_stressed:.2f}**, "
        f"pushing expected loss to **₹{stats_stressed['EL']:,.0f}** and VaR({var_confidence}%) to "
        f"**₹{stats_stressed['VaR']:,.0f}**."
    )
    shield_holds = stats_stressed_mitigated["EL"] < stats_stressed["EL"]
    if shield_holds:
        st.success(
            f"**Shield performance under stress:** even in the anomaly scenario, the mitigated portfolio's "
            f"total cost (₹{stats_stressed_mitigated['EL']:,.0f}) stays below the unmitigated stressed loss — "
            f"the shield still adds value in a bad year."
        )
    else:
        st.warning(
            f"**Shield performance under stress:** under the anomaly scenario, mitigated total cost "
            f"(₹{stats_stressed_mitigated['EL']:,.0f}) approaches or exceeds the unmitigated stressed loss "
            f"(₹{stats_stressed['EL']:,.0f}) — the shield's benefit erodes when contagion correlation spikes."
        )
    st.caption(f"Regional diversification: farms are spread across **{num_regions}** independent region(s). "
               "More regions with uncorrelated weather/disease exposure meaningfully lower tail risk — "
               "see the diversification chart below.")

st.markdown("---")

# ==================================================================
# VISUALS
# ==================================================================
viz1, viz2 = st.columns(2)

with viz1:
    st.subheader("Expected Loss vs. Tail Risk by Scenario")
    scenario_names = ["Baseline", "Mitigated", "Stressed\n(no shield)", "Stressed\n(with shield)"]
    els = [stats_baseline["EL"], stats_mitigated["EL"], stats_stressed["EL"], stats_stressed_mitigated["EL"]]
    vars_ = [stats_baseline["VaR"], stats_mitigated["VaR"], stats_stressed["VaR"], stats_stressed_mitigated["VaR"]]

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(scenario_names))
    width = 0.35
    ax1.bar(x - width/2, els, width, label="Expected Loss", color="#00c0f2")
    ax1.bar(x + width/2, vars_, width, label=f"VaR({var_confidence}%)", color="#ff4b4b")
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, fontsize=8)
    ax1.set_ylabel("Amount (₹)")
    ax1.ticklabel_format(style="plain", axis="y")
    ax1.legend(fontsize=8)
    st.pyplot(fig1)

with viz2:
    st.subheader("Simulated Loss Distribution: Baseline vs. Mitigated")
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(stats_baseline["losses"], bins=40, alpha=0.5, label="Baseline", color="#ff4b4b", density=True)
    ax2.hist(stats_mitigated["losses"], bins=40, alpha=0.5, label="Mitigated", color="#00c0f2", density=True)
    ax2.axvline(stats_baseline["VaR"] - fixed_cost_baseline, color="#ff4b4b", linestyle="--", linewidth=1)
    ax2.axvline(stats_mitigated["VaR"] - fixed_cost_mitigated, color="#00c0f2", linestyle="--", linewidth=1)
    ax2.set_xlabel("Portfolio Loss (₹, excluding fixed costs)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=8)
    st.pyplot(fig2)

st.subheader("Diversification Benefit: Tail Risk vs. Number of Regions")
region_options = sorted(set([1, 2, 3, 5, 8, 12, 16, 20, num_regions]))
region_options = [r for r in region_options if r <= num_farms]
div_var = []
for r in region_options:
    region_id_temp = np.arange(num_farms) % r
    losses_r = simulate_losses(
        p_baseline_farms, rho_infection, max(500, num_simulations // 4),
        region_id_arr=region_id_temp, n_regions=r,
    )
    div_var.append(np.percentile(losses_r, var_confidence))

fig3, ax3 = plt.subplots(figsize=(12, 3.5))
ax3.plot(region_options, div_var, marker="o", color="#31333f")
ax3.axvline(num_regions, color="#ff4b4b", linestyle="--", label=f"Current setting: {num_regions} regions")
ax3.set_xlabel("Number of Independent Regions")
ax3.set_ylabel(f"Portfolio VaR({var_confidence}%) (₹)")
ax3.legend(fontsize=8)
st.pyplot(fig3)

st.subheader("Risk Factor Summary")
factor_df = pd.DataFrame({
    "Factor": [
        "Baseline Infection Probability", "Contagion Correlation (ρ)", "Loss Given Infection (raw)",
        "Effective LGI (after detection/quarantine)", "Bio-Shield Adoption", "Bio-Shield Efficacy",
        "Insurance Coverage", "Climate Stress Multiplier", "Stress Correlation Add-on",
    ],
    "Value": [
        f"{p_baseline*100:.1f}%", f"{rho_infection:.2f}", f"{lgi_base*100:.1f}%",
        f"{lgi_effective*100:.1f}%", f"{shield_adoption_pct*100:.0f}% ({num_adopters}/{num_farms} farms)",
        f"{shield_efficacy_pct*100:.0f}%", f"{insurance_coverage_pct*100:.0f}%",
        f"{p_stress_multiplier:.1f}×", f"+{rho_stress_add:.2f}",
    ],
})
st.dataframe(factor_df, hide_index=True, use_container_width=True)

st.caption(
    "⚠️ **Model limitations:** infection probabilities, correlation, and stress multipliers are "
    "illustrative anchors, not fitted from historical outbreak data. The single-factor-per-region "
    "contagion model is a simplification of real epidemiological spread (which also depends on "
    "distance, wind/water vectors, and vector population dynamics). Validate against agronomic "
    "and claims history before using this for real underwriting or capital decisions."
)
