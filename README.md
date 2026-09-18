# Circumstellar Habitable Zone Simulator

This project was presented as submission for the YPAE second edition Simathon (September 2026).
Presents an interactive Python simulation modeling planetary orbital mechanics and circumstellar habitable zones, backed by physical thermodynamic equations.

## Overview
This project simulates gravitational orbits using numerical integration and dynamically calculates habitable zones based on stellar properties and atmospheric modeling. Built with Python and Taichi for performance with AI assistance (Antigravity, Gemini).

## Features
* **Orbital Mechanics:** Simulates gravitational interactions using a symplectic Velocity Verlet integrator.
* **Habitable Zone Calculation:** Computes inner and outer circumstellar boundaries dynamically based on established astrophysical models.
* **Interactive Controls:** Pause, reset, and adjust simulation parameters on the fly.

---

## Nature of the Model & Formal Considerations

Before detailing the mathematical derivations, it is necessary to establish the framework through the principles of mathematical modeling. The system is treated as a zero-dimensional lumped-parameter model designed to yield qualitative and steady-state solutions rather than transient meteorological forecasts.

The foundational physical assumptions and formal modeling considerations include:

1. **Global Average Surface Temperature (Zeroth-Order Spatial Lumping):**  
   * The temperature $T$ solved in this model represents the global average surface temperature of the planet.
   * Spatially distributed partial differential equations for heat transfer are lumped into an algebraic global balance. Rapid planetary rotation, atmospheric circulation, and oceanic heat transport redistribute thermal energy efficiently, allowing a zero-dimensional approach to approximate long-term global thermal equilibrium.

2. **Single-Layer Atmosphere Approximation:**
   * The complex multi-layered vertical atmosphere of the planet is reduced to a single isothermal atmospheric layer characterized by a uniform absorptivity and emissivity $\varepsilon$.
   * This classic radiative-convective greenhouse abstraction captures the dominant physical mechanism of atmospheric warming—absorbing outgoing terrestrial longwave radiation and re-radiating it isotropically—while keeping the analytical equations tractable and soluble.

3. **Constant Planetary Albedo ($\alpha$):**
   * The bond albedo is modeled as a time-invariant fraction reflecting incoming stellar shortwave radiation.
   * It establishes a clean, robust first-order baseline for planetary reflectivity across standard stellar spectra before introducing higher-order non-linearities such as cloud-feedback dynamics.

4. **Negligible Geothermal and Internal Heat Flux:**
   * Internal planetary heat sources (such as radiogenic decay and primordial core cooling) are assumed to be negligible compared to the radiative flux intercepted from the parent star ($Q_{\text{internal}} \ll Q_{\text{stellar}}$).
   * For terrestrial planets located within circumstellar habitable zones, stellar insolation dominates the surface energy budget by orders of magnitude, rendering internal thermal contributions mathematically secondary.

5. **Steady-State Equilibrium ($\frac{dT}{dt} = 0$):**
   * The governing differential equations are evaluated under steady-state conditions, setting time derivatives to zero to solve for long-term energetic balance.
   * The model intentionally filters out short-term transient noise (diurnal cycles, seasonal weather shifts) to focus on macro-scale qualitative solutions. Its primary objective is to map out asymptotic stability thresholds and critical bifurcation boundaries (such as the moist and runaway greenhouse tipping points) rather than tracking instantaneous meteorological states.

## Definition and Limits of the Habitable Zone

The stellar or circumstellar habitable zone (HZ) corresponds to the circular region around a star within which a rocky planet with favorable atmospheric conditions can sustain permanent liquid water bodies on its surface[cite: 2]. Because the central star is the primary energy source for a planet with Earth-like characteristics and therefore the determining factor of water's physical state, it is logical to assume a correlation between its properties (mass, luminosity, evolutionary stage, spectral class, etc.) and the location of the HZ boundaries.

Numerous models exist for calculating the HZ, depending on the bounding conditions of the region. Generally, these consider, based on stellar properties, where an Earth-like planet could harbor liquid water bodies. For example, conservative limits correspond to the runaway greenhouse and maximum greenhouse limits—that is, the effect of atmospheric conditions on planetary temperature[cite: 2]. These establish that, for the Solar System, the inner limit is located at 0.95 AU and the outer limit at 1.67 AU[cite: 2]. Optimistic limits also exist, which are estimates based on observations and hypotheses regarding the possibility that Mars harbored liquid water on its surface billions of years ago and theories on water loss via evaporation on Venus. These establish, for the Solar System, an inner limit of 0.75 AU and an outer limit of 1.77 AU.

### Specific Boundary Criteria Used in This Model:
* **Outer Limit:** Corresponds to the distance where the planetary temperature equals the freezing point of water at 273.15 K.
* **Inner Limit:** Adapts the considerations described by Ramirez (2018): it corresponds to the moist greenhouse effect point, when the water vapor mixing ratio above the tropopause—understood as the point of minimum atmospheric temperature where rising air cannot ascend further—prevents water vapor convection due to low temperatures. This would cause the eventual total evaporation of surface water bodies, escaping into space[cite: 2]. This can occur at an estimated planetary temperature of 340 K[cite: 2]. Although the boiling point of water could also be used, these considerations allow incorporating atmospheric factors for a more realistic model.

---

## Mathematical Habitable Zone Model

The simulation dynamically computes the habitable zone boundaries using a single-layer atmosphere model governed by the following energy balance equations.

### 1. Energy Balance Equations
The total stellar luminosity emitted by a star with radius $R_\odot$ and effective temperature $T_\odot$ is given by:
$$L_\odot = 4\pi R_\odot^2 \sigma T_\odot^4$$

At an orbital distance $d$, the surface of the planet receives radiation over its cross-sectional area $\pi r_e^2$. Considering planetary albedo $\alpha$ and atmospheric emissivity/absorptivity $\varepsilon$, the energy balance yields:

$$\text{Energy In} = 4\pi R_\odot^2 \sigma T_\odot^4 \left( \frac{\pi r_e^2}{4\pi d^2} \right) (1 - \alpha) + 4\pi r_e^2 \sigma T_a^4 \varepsilon$$

$$\text{Energy Out} = 4\pi r_e^2 \sigma T^4$$

Equating $\text{Energy In} = \text{Energy Out}$ and simplifying factors of $4\pi \sigma$:
$$\frac{R_\odot^2 T_\odot^4 r_e^2 (1 - \alpha)}{4 d^2} + r_e^2 T_a^4 \varepsilon = r_e^2 T^4$$

### 2. Atmospheric Equilibrium
Applying Kirchhoff's Law of Thermal Radiation ($\varepsilon = A$), the single atmospheric layer absorbs thermal radiation from the surface and re-radiates isotropically both upwards and downwards:
$$2 r_e^2 T_a^4 \varepsilon = r_e^2 T^4 \varepsilon \implies r_e^2 T_a^4 = \frac{1}{2} r_e^2 T^4$$

### 3. Surface Temperature $T(d)$
Substituting the atmospheric expression into the surface energy balance equation and isolating $T^4$:
$$T(d) = \sqrt[4]{\frac{1 - \alpha}{4 - 2\varepsilon}} \sqrt{\frac{R_\odot}{d}} T_\odot$$

### 4. Habitable Zone Boundary Distance $d$
Solving for the orbital distance $d$ as a function of target surface temperature $T$:
$$d = \frac{T_\odot^2 R_\odot}{2 T^2} \sqrt{\frac{2(1 - \alpha)}{2 - \varepsilon}}$$

### Reference Parameters (Solar Baseline)
$$\text{Habitable Zone Distance Formula: } d = \frac{R_\star T_\star^2}{2 T_{\text{planet}}^2} \sqrt{\frac{2(1 - \alpha)}{2 - \varepsilon}}$$

* **$T_\star$**: Stellar Temperature in Kelvin (Sun = 5778 K)
* **$R_\star$**: Stellar Radius in Solar Radii (Sun = $1.0 \, R_\odot \approx 0.00465 \text{ AU}$)
* **$T_{\text{planet}}$**: Surface Temperature in Kelvin (Liquid Water limits: 273.15 K to 340 K)
* **$\alpha$**: Planetary Albedo (Earth $\approx 0.306$)
* **$\varepsilon$**: Atmospheric Emissivity / Greenhouse Effect (Earth $\approx 0.77$)

---

## Interactive Controls & Visualization

### Keyboard & Mouse Shortcuts
* **`SPACE`**: Pause / Resume simulation
* **`R`**: Reset to Earth-Sun default baseline
* **`C`**: Cycle camera center (Origin / Star / Planet)
* **`G`**: Toggle AU distance reference grid
* **`H`**: Toggle Habitable Zone volumetric mist
* **`D`**: Toggle Circumstellar Dust & Asteroid Belt
* **`T`**: Toggle Planetary Orbit Trail
* **`L`**: Toggle Lagrange Point holographic reticles
* **`+` / `-`**: Zoom In / Zoom Out
* **`[` / `]`**: Slow down / Speed up time step
* **Left-Click Drag**: Pan camera in AU space
* **Right-Click**: Relocate planet to cursor and set circular orbit

### Emergent Phenomena Visible
* Resonant clearing of dust feeding zone along the planet's orbit.
* Stable Trojan asteroid swarms trapped around L4 and L5 Lagrange points.
* Thermal color evolution: planet roasts at perihelion and freezes at aphelion.
* Stellar wobble / reflex motion around system barycenter when planet mass is high.

---

## Requirements & Installation
1. Install dependencies:
   ```bash
   pip install taichi numpy
   
   
