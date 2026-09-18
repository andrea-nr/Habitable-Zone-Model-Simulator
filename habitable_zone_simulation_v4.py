"""
Habitable Zone & Planetary Orbit Simulator
=========================================
A physically rigorous, cinematic simulation of circumstellar habitable zones,
planetary orbital mechanics, and thermal equilibria using Taichi Lang.

Physics Model:
  Habitable Zone Radial Distance:
    d = (R_star * T_star^2) / (2 * T_planet^2) * sqrt(2 * (1 - alpha) / (2 - epsilon))
  where:
    T_star    = Stellar effective surface temperature [Kelvin]
    R_star    = Stellar radius [AU]
    T_planet  = Planetary equilibrium surface temperature [Kelvin]
    alpha     = Planetary Bond albedo [dimensionless]
    epsilon   = Atmospheric greenhouse emissivity [dimensionless]

  Habitable Boundaries:
    Inner Edge (Moist Greenhouse / Water Loss): T = 340.0 K
    Outer Edge (Maximum Greenhouse / Freezing): T = 273.15 K

Integrator:
  Symplectic Velocity Verlet (conserving phase-space volume & orbital energy).
  Units: Astronomical Units [AU], Solar Masses [M_sun], Years [yr].
  In these units: Gravitational constant G = 4 * pi^2.
"""

import math
import sys
import numpy as np
import taichi as ti

# ==============================================================================
# 1. TAICHI INITIALIZATION & GUARDRAILS
# ==============================================================================
try:
    ti.init(arch=ti.gpu)
except Exception:
    ti.init(arch=ti.cpu)
    print("Running on CPU, expect lower FPS. Lower the body count if it crawls.")

# ==============================================================================
# 2. PHYSICAL CONSTANTS & CONVERSIONS (ASTRONOMICAL UNITS)
# ==============================================================================
# Gravitational constant in AU^3 / (M_sun * yr^2):
# Since Earth orbits at 1 AU in 1 yr around a 1 M_sun star: v = 2*pi, a = v^2/r = 4*pi^2 = G*M/r^2 => G = 4*pi^2.
G_CONST = 4.0 * math.pi * math.pi          # ~39.4784176 AU^3 / (M_sun * yr^2)

# Solar conversions:
SOLAR_RADIUS_TO_AU = 0.00465047            # 1 R_sun = 6.957e8 m / 1.4959787e11 m
EARTH_MASS_TO_SOLAR_MASS = 3.003e-6        # 1 M_earth in M_sun (5.972e24 kg / 1.989e30 kg)
AU_TO_KM = 1.4959787e8                     # 1 AU in km
YR_TO_SEC = 365.25 * 86400.0               # 1 Julian year in seconds
AU_PER_YR_TO_KM_S = AU_TO_KM / YR_TO_SEC   # 1 AU/yr ~ 4.74047 km/s

# Temperature boundaries for Habitable Zone (Liquid Water):
T_INNER_HZ = 340.0                         # Moist Greenhouse runaway threshold [K]
T_OUTER_HZ = 273.15                        # Freezing point of liquid water [K]

# Stellar Evolution Stages (Sun-like star lifecycle sequence)
evo_stages = [
        {"name": "1. Zero-Age Main Sequence (ZAMS)", "t_star": 5600.0, "r_star": 0.9, "a_planet": 1.0, "e_planet": 0.01, "m_earth": 1.0, "alpha": 0.306, "epsilon": 0.77, "zoom": 2.4},
        {"name": "2. Mid Life / Present Sun",        "t_star": 5778.0, "r_star": 1.0, "a_planet": 1.0, "e_planet": 0.0167, "m_earth": 1.0, "alpha": 0.306, "epsilon": 0.77, "zoom": 2.4},
        {"name": "3. Late Main Sequence (Bright Sun)", "t_star": 5850.0, "r_star": 1.2, "a_planet": 1.0, "e_planet": 0.0167, "m_earth": 1.0, "alpha": 0.306, "epsilon": 0.77, "zoom": 2.5},
        {"name": "4. Subgiant Branch (Expanding)",     "t_star": 5200.0, "r_star": 2.3, "a_planet": 1.5, "e_planet": 0.02, "m_earth": 1.0, "alpha": 0.306, "epsilon": 0.77, "zoom": 3.5},
        {"name": "5. Red Giant Phase (HZ Outward)",    "t_star": 3500.0, "r_star": 15.0, "a_planet": 3.0, "e_planet": 0.05, "m_earth": 1.0, "alpha": 0.306, "epsilon": 0.77, "zoom": 6.0},
        {"name": "6. White Dwarf Remnant (Cooling)",   "t_star": 12000.0, "r_star": 0.02, "a_planet": 1.0, "e_planet": 0.01, "m_earth": 1.0, "alpha": 0.306, "epsilon": 0.77, "zoom": 2.4}
    ]

# Numerical softening to prevent gravitational singularity:
GRAV_EPS = 0.005                           # Epsilon in AU (~750,000 km)

# Particle Counts:
N_DUST = 2000                             # Circumstellar dust & Trojan asteroid particles
N_HZ_GLOW = 1500                          # Volumetric Habitable Zone glow particles
N_CORONA = 250                             # Stellar corona flare particles
N_TRAIL = 1200                             # Orbit trail history points
N_RING_SEGS = 64                         # Smooth ring segments for HZ boundaries
N_GRID_RINGS = 6                           # Reference AU distance rings
N_GRID_SEGS = 32                         # Segments per grid ring

# Window Dimensions:
WIN_W = 1280
WIN_H = 720
ASPECT = float(WIN_W) / float(WIN_H)

# ==============================================================================
# 3. TAICHI DATA FIELDS (PER-BODY AND ENVIRONMENT STATE)
# ==============================================================================
# Major celestial bodies: Index 0 = Star, Index 1 = Planet
pos_bodies = ti.Vector.field(2, dtype=ti.f32, shape=2)
vel_bodies = ti.Vector.field(2, dtype=ti.f32, shape=2)
acc_bodies = ti.Vector.field(2, dtype=ti.f32, shape=2)
mass_bodies = ti.field(dtype=ti.f32, shape=2)
color_bodies = ti.Vector.field(3, dtype=ti.f32, shape=2)
radius_bodies = ti.field(dtype=ti.f32, shape=2)

# Planetary diagnostic scalars:
planet_temp = ti.field(dtype=ti.f32, shape=())
planet_dist = ti.field(dtype=ti.f32, shape=())
hz_inner_dist = ti.field(dtype=ti.f32, shape=())
hz_outer_dist = ti.field(dtype=ti.f32, shape=())

# Circumstellar dust & Trojan asteroid field (Restricted 3-body simulation):
pos_dust = ti.Vector.field(2, dtype=ti.f32, shape=N_DUST)
vel_dust = ti.Vector.field(2, dtype=ti.f32, shape=N_DUST)
acc_dust = ti.Vector.field(2, dtype=ti.f32, shape=N_DUST)
color_dust = ti.Vector.field(3, dtype=ti.f32, shape=N_DUST)

# Habitable Zone volumetric luminous mist:
pos_hz = ti.Vector.field(2, dtype=ti.f32, shape=N_HZ_GLOW)
color_hz = ti.Vector.field(3, dtype=ti.f32, shape=N_HZ_GLOW)
rand_hz = ti.Vector.field(2, dtype=ti.f32, shape=N_HZ_GLOW)  # (normalized radius fraction, orbital angle)

# Stellar corona particles:
pos_corona = ti.Vector.field(2, dtype=ti.f32, shape=N_CORONA)
color_corona = ti.Vector.field(3, dtype=ti.f32, shape=N_CORONA)
phase_corona = ti.Vector.field(3, dtype=ti.f32, shape=N_CORONA) # (base_angle, speed, radial_offset)

# Planet Orbit Trail:
trail_pos = ti.Vector.field(2, dtype=ti.f32, shape=N_TRAIL)
trail_color = ti.Vector.field(3, dtype=ti.f32, shape=N_TRAIL)
trail_head = ti.field(dtype=ti.i32, shape=())
trail_len = ti.field(dtype=ti.i32, shape=())

# Line vertex buffers for canvas.lines:
# 1. Inner HZ boundary: N_RING_SEGS * 2 vertices
inner_ring_verts = ti.Vector.field(2, dtype=ti.f32, shape=N_RING_SEGS * 2)
inner_ring_color = ti.Vector.field(3, dtype=ti.f32, shape=N_RING_SEGS * 2)

# 2. Outer HZ boundary: N_RING_SEGS * 2 vertices
outer_ring_verts = ti.Vector.field(2, dtype=ti.f32, shape=N_RING_SEGS * 2)
outer_ring_color = ti.Vector.field(3, dtype=ti.f32, shape=N_RING_SEGS * 2)

# 3. Grid distance rings: N_GRID_RINGS * N_GRID_SEGS * 2 vertices
grid_verts = ti.Vector.field(2, dtype=ti.f32, shape=N_GRID_RINGS * N_GRID_SEGS * 2)
grid_color = ti.Vector.field(3, dtype=ti.f32, shape=N_GRID_RINGS * N_GRID_SEGS * 2)

# 4. Trail lines: (N_TRAIL - 1) * 2 vertices
trail_line_verts = ti.Vector.field(2, dtype=ti.f32, shape=(N_TRAIL - 1) * 2)
trail_line_color = ti.Vector.field(3, dtype=ti.f32, shape=(N_TRAIL - 1) * 2)

# 5. Diagnostic vector arrows & markers (velocity vector, star-planet tether):
diag_lines_verts = ti.Vector.field(2, dtype=ti.f32, shape=128)
diag_lines_color = ti.Vector.field(3, dtype=ti.f32, shape=128)
diag_lines_count = ti.field(dtype=ti.i32, shape=())

# Lagrange markers circles:
lagrange_pos = ti.Vector.field(2, dtype=ti.f32, shape=5)
lagrange_color = ti.Vector.field(3, dtype=ti.f32, shape=5)

# Render canvas positions (mapped to screen space [0, 1]):
screen_dust = ti.Vector.field(2, dtype=ti.f32, shape=N_DUST)
screen_hz = ti.Vector.field(2, dtype=ti.f32, shape=N_HZ_GLOW)
screen_corona = ti.Vector.field(2, dtype=ti.f32, shape=N_CORONA)
screen_bodies = ti.Vector.field(2, dtype=ti.f32, shape=2)
screen_lagrange = ti.Vector.field(2, dtype=ti.f32, shape=5)

star_halo_pos = ti.Vector.field(2, dtype=ti.f32, shape=1)
star_halo_col = ti.Vector.field(3, dtype=ti.f32, shape=1)
# Temperature scale / legend
N_TEMP_SCALE = 100
temp_scale_pos = ti.Vector.field(2, dtype=ti.f32, shape=N_TEMP_SCALE)
temp_scale_col = ti.Vector.field(3, dtype=ti.f32, shape=N_TEMP_SCALE)

planet_pt = ti.Vector.field(2, dtype=ti.f32, shape=1)
planet_col = ti.Vector.field(3, dtype=ti.f32, shape=1)
planet_halo_col = ti.Vector.field(3, dtype=ti.f32, shape=1)

# ==============================================================================
# 4. COLOR & THERMAL EQUILIBRIUM MATHEMATICAL FUNCTIONS
# ==============================================================================
@ti.func
def compute_hz_radius(t_planet: ti.f32, t_star: ti.f32, r_star_au: ti.f32, alpha: ti.f32, epsilon: ti.f32) -> ti.f32:
    """
    Computes circumstellar orbital radius d for a planet at temperature t_planet.
    d = (R_star * T_star^2) / (2 * T_planet^2) * sqrt( 2*(1 - alpha) / (2 - epsilon) )
    """
    eps_clamped = ti.min(1.99, ti.max(0.0, epsilon))
    rad_ratio = ti.sqrt(ti.max(1e-6, 2.0 * (1.0 - alpha) / (2.0 - eps_clamped)))
    t_ratio_sq = (t_star * t_star) / (2.0 * t_planet * t_planet + 1e-4)
    return r_star_au * t_ratio_sq * rad_ratio


@ti.func
def compute_planet_temp(dist: ti.f32, t_star: ti.f32, r_star_au: ti.f32, alpha: ti.f32, epsilon: ti.f32) -> ti.f32:
    """
    Inverts the radiative equilibrium formula to obtain planetary surface temperature T:
    T = T_star * sqrt(R_star / (2 * dist)) * ( 2*(1 - alpha) / (2 - epsilon) )^(1/4)
    """
    eps_clamped = ti.min(1.99, ti.max(0.0, epsilon))
    factor = ti.max(1e-6, 2.0 * (1.0 - alpha) / (2.0 - eps_clamped))
    greenhouse_boost = ti.sqrt(ti.sqrt(factor))
    d_safe = ti.max(1e-3, dist)
    geom_factor = ti.sqrt(r_star_au / (2.0 * d_safe))
    return t_star * geom_factor * greenhouse_boost


@ti.func
def temperature_to_color(temp_k: ti.f32) -> ti.template():
    """
    Maps physical temperature [Kelvin] to a continuous, aesthetically vibrant palette:
      - Frozen (T < 273.15 K): Crystalline Glacial Cyan to Deep Cosmic Violet
      - Habitable (273.15 K <= T <= 340 K): Oceanic Aquamarine to Biosphere Emerald
      - Moist / Runaway Greenhouse (T > 340 K): Scorching Solar Amber to Volcanic Crimson
    """
    col = ti.Vector([0.5, 0.5, 0.5])
    if temp_k < 200.0:
        t = ti.max(0.0, temp_k / 200.0)
        col = ti.Vector([0.15 + 0.15 * t, 0.2 + 0.3 * t, 0.65 + 0.35 * t])
    elif temp_k < 273.15:
        t = (temp_k - 200.0) / 73.15
        col = ti.Vector([0.3 - 0.15 * t, 0.5 + 0.35 * t, 1.0 - 0.1 * t])
    elif temp_k < 305.0:
        t = (temp_k - 273.15) / 31.85
        col = ti.Vector([0.05 + 0.1 * t, 0.85 + 0.15 * t, 0.9 - 0.25 * t])
    elif temp_k <= 340.0:
        t = (temp_k - 305.0) / 35.0
        col = ti.Vector([0.15 + 0.7 * t, 1.0 - 0.2 * t, 0.65 - 0.55 * t])
    elif temp_k < 450.0:
        t = (temp_k - 340.0) / 110.0
        col = ti.Vector([0.85 + 0.15 * t, 0.8 - 0.45 * t, 0.1])
    else:
        t = ti.min(1.0, (temp_k - 450.0) / 500.0)
        col = ti.Vector([1.0, 0.35 - 0.25 * t, 0.05])
    return col


@ti.func
def star_blackbody_color(t_star: ti.f32) -> ti.template():
    """
    Approximates Planckian locus RGB color for a star of temperature T_star [K].
    Cool M-dwarf (2500K) -> G-dwarf (5800K) -> Hot A/B-star (10000K).
    """
    col = ti.Vector([1.0, 1.0, 1.0])
    if t_star < 3500.0:
        t = (t_star - 2000.0) / 1500.0
        col = ti.Vector([1.0, 0.3 + 0.3 * t, 0.05 + 0.1 * t])
    elif t_star < 5200.0:
        t = (t_star - 3500.0) / 1700.0
        col = ti.Vector([1.0, 0.6 + 0.25 * t, 0.15 + 0.45 * t])
    elif t_star < 6500.0:
        t = (t_star - 5200.0) / 1300.0
        col = ti.Vector([1.0, 0.85 + 0.1 * t, 0.6 + 0.3 * t])
    elif t_star < 8000.0:
        t = (t_star - 6500.0) / 1500.0
        col = ti.Vector([0.95 + 0.05 * t, 0.95 + 0.05 * t, 0.9 + 0.1 * t])
    else:
        t = ti.min(1.0, (t_star - 8000.0) / 4000.0)
        col = ti.Vector([0.85 - 0.1 * t, 0.9 + 0.05 * t, 1.0])
    return col


# ==============================================================================
# 5. INITIALIZATION KERNELS
# ==============================================================================
@ti.kernel
def init_simulation(t_star: ti.f32, r_star_solar: ti.f32, a_planet: ti.f32, e_planet: ti.f32,
                    m_planet_earth: ti.f32, alpha: ti.f32, epsilon: ti.f32):
    r_star_au = r_star_solar * SOLAR_RADIUS_TO_AU
    m_star = 0.09
    m_planet = m_planet_earth * EARTH_MASS_TO_SOLAR_MASS
    mass_bodies[0] = m_star
    mass_bodies[1] = m_planet

    # Keplerian perihelion setup for Planet:
    r_peri = a_planet * (1.0 - e_planet)
    v_peri = ti.sqrt(G_CONST * (m_star + m_planet) / a_planet * (1.0 + e_planet) / (1.0 - e_planet + 1e-6))

    # Center-of-mass barycentric coordinates:
    pos_bodies[0] = ti.Vector([-m_planet / (m_star + m_planet) * r_peri, 0.0])
    vel_bodies[0] = ti.Vector([0.0, -m_planet / (m_star + m_planet) * v_peri])
    acc_bodies[0] = ti.Vector([0.0, 0.0])

    pos_bodies[1] = ti.Vector([m_star / (m_star + m_planet) * r_peri, 0.0])
    vel_bodies[1] = ti.Vector([0.0, m_star / (m_star + m_planet) * v_peri])
    acc_bodies[1] = ti.Vector([0.0, 0.0])

    color_bodies[0] = star_blackbody_color(t_star)
    radius_bodies[0] = ti.max(0.012, 0.015 * r_star_solar)
    radius_bodies[1] = ti.max(0.005, 0.006 * ti.pow(m_planet_earth, 0.2))

    # Habitable Zone boundaries:
    d_inner = compute_hz_radius(T_INNER_HZ, t_star, r_star_au, alpha, epsilon)
    d_outer = compute_hz_radius(T_OUTER_HZ, t_star, r_star_au, alpha, epsilon)
    hz_inner_dist[None] = d_inner
    hz_outer_dist[None] = d_outer

    # Seed Circumstellar Dust & Trojan Asteroid Field:
    for i in range(N_DUST):
        r_dust = 0.0
        theta = 0.0
        if i < 16000:
            # Protoplanetary disk: r distributed from 0.3 to 3.5 AU
            u = ti.random(ti.f32)
            r_dust = 0.3 + 3.2 * ti.sqrt(u)
            theta = ti.random(ti.f32) * 2.0 * math.pi
        elif i < 20000:
            # L4 Trojan asteroid swarm (60 degrees leading planet):
            r_dust = a_planet + (ti.random(ti.f32) - 0.5) * 0.35 * a_planet
            theta = math.pi / 3.0 + (ti.random(ti.f32) - 0.5) * 0.45
        else:
            # L5 Greek asteroid swarm (60 degrees trailing planet):
            r_dust = a_planet + (ti.random(ti.f32) - 0.5) * 0.35 * a_planet
            theta = -math.pi / 3.0 + (ti.random(ti.f32) - 0.5) * 0.45

        # Circular Keplerian velocity with slight random dispersion:
        v_circ = ti.sqrt(G_CONST * m_star / (r_dust + 1e-4))
        ecc_pert = (ti.random(ti.f32) - 0.5) * 0.04
        v_circ *= (1.0 + ecc_pert)

        pos_dust[i] = ti.Vector([r_dust * ti.cos(theta), r_dust * ti.sin(theta)])
        vel_dust[i] = ti.Vector([-v_circ * ti.sin(theta), v_circ * ti.cos(theta)])
        acc_dust[i] = ti.Vector([0.0, 0.0])

        t_p = compute_planet_temp(r_dust, t_star, r_star_au, alpha, epsilon)
        color_dust[i] = temperature_to_color(t_p)

    # Seed Habitable Zone Volumetric Mist:
    for i in range(N_HZ_GLOW):
        rand_hz[i] = ti.Vector([ti.random(ti.f32), ti.random(ti.f32) * 2.0 * math.pi])

    # Seed Stellar Corona particles:
    for i in range(N_CORONA):
        phase_corona[i] = ti.Vector([
            ti.random(ti.f32) * 2.0 * math.pi,
            0.5 + ti.random(ti.f32) * 1.5,
            ti.random(ti.f32)
        ])

    trail_head[None] = 0
    trail_len[None] = 0


@ti.kernel
def reset_planet_kepler(t_star: ti.f32, r_star_solar: ti.f32, a_planet: ti.f32, e_planet: ti.f32,
                        m_planet_earth: ti.f32, alpha: ti.f32, epsilon: ti.f32):
    m_star = mass_bodies[0]
    m_planet = m_planet_earth * EARTH_MASS_TO_SOLAR_MASS
    mass_bodies[1] = m_planet

    r_peri = a_planet * (1.0 - e_planet)
    v_peri = ti.sqrt(G_CONST * (m_star + m_planet) / a_planet * (1.0 + e_planet) / (1.0 - e_planet + 1e-6))

    star_pos = pos_bodies[0]
    pos_bodies[1] = star_pos + ti.Vector([r_peri, 0.0])
    vel_bodies[1] = vel_bodies[0] + ti.Vector([0.0, v_peri])

    trail_head[None] = 0
    trail_len[None] = 0


# ==============================================================================
# 6. SYMPLECTIC VELOCITY VERLET INTEGRATION KERNEL
# ==============================================================================
@ti.kernel
def compute_accelerations(t_star: ti.f32, r_star_solar: ti.f32, alpha: ti.f32, epsilon: ti.f32):
    m_star = mass_bodies[0]
    m_planet = mass_bodies[1]
    p_star = pos_bodies[0]
    p_planet = pos_bodies[1]

    # Mutual Star-Planet gravity:
    r_vec = p_planet - p_star
    r_sq = r_vec.dot(r_vec) + GRAV_EPS * GRAV_EPS
    r_dist = ti.sqrt(r_sq)
    force_mag = G_CONST / (r_sq * r_dist)

    acc_bodies[0] = force_mag * m_planet * r_vec
    acc_bodies[1] = -force_mag * m_star * r_vec

    # Dust particles gravitation (Restricted 3-body gravitational field):
    for i in range(N_DUST):
        p_d = pos_dust[i]

        r_ds = p_d - p_star
        r_ds_sq = r_ds.dot(r_ds) + GRAV_EPS * GRAV_EPS
        f_s = -G_CONST * m_star / (r_ds_sq * ti.sqrt(r_ds_sq))

        r_dp = p_d - p_planet
        r_dp_sq = r_dp.dot(r_dp) + GRAV_EPS * GRAV_EPS
        f_p = -G_CONST * m_planet / (r_dp_sq * ti.sqrt(r_dp_sq))

        acc_dust[i] = f_s * r_ds + f_p * r_dp


@ti.kernel
def verlet_step_positions(dt: ti.f32):
    for b in range(2):
        vel_bodies[b] += 0.5 * acc_bodies[b] * dt
        pos_bodies[b] += vel_bodies[b] * dt

    for i in range(N_DUST):
        vel_dust[i] += 0.5 * acc_dust[i] * dt
        pos_dust[i] += vel_dust[i] * dt



@ti.kernel
def verlet_step_velocities(dt: ti.f32):
    for b in range(2):
        vel_bodies[b] += 0.5 * acc_bodies[b] * dt

    p_star = pos_bodies[0]

    for i in range(N_DUST):
        vel_dust[i] += 0.5 * acc_dust[i] * dt

        # Distance of dust particle from the star
        d_vec = pos_dust[i] - p_star
        d_r = ti.sqrt(d_vec.dot(d_vec))

        # Recycling boundary: keep simulation alive forever
        if d_r > 5.5 or d_r < 0.08:
            new_r = 0.5 + 3.0 * ti.sqrt(ti.random(ti.f32))
            new_th = ti.random(ti.f32) * 2.0 * math.pi
            v_c = ti.sqrt(G_CONST * mass_bodies[0] / new_r)
            pos_dust[i] = p_star + ti.Vector([
                new_r * ti.cos(new_th),
                new_r * ti.sin(new_th)
            ])
            vel_dust[i] = vel_bodies[0] + ti.Vector([
                -v_c * ti.sin(new_th),
                v_c * ti.cos(new_th)
            ])

# ==============================================================================
# 7. VISUAL ELEMENTS, CORONA, TRAILS & RETICLES UPDATE KERNELS
# ==============================================================================
@ti.kernel
def update_dust_colors(
    t_star: ti.f32,
    r_star_solar: ti.f32,
    alpha: ti.f32,
    epsilon: ti.f32
):
    r_star_au = r_star_solar * SOLAR_RADIUS_TO_AU
    p_star = pos_bodies[0]

    for i in range(N_DUST):
        d = (pos_dust[i] - p_star).norm()
        temp = compute_planet_temp(
            d,
            t_star,
            r_star_au,
            alpha,
            epsilon
        )
        color_dust[i] = temperature_to_color(temp)

@ti.kernel
def update_temperature_scale():
    for i in range(N_TEMP_SCALE):
        # Temperature range represented by the legend: 150 K to 600 K
        temp = 150.0 + 450.0 * float(i) / float(N_TEMP_SCALE - 1)

        # Horizontal gradient positioned near the bottom-right of the screen
        x = 0.78 + 0.18 * float(i) / float(N_TEMP_SCALE - 1)
        y = 0.08

        temp_scale_pos[i] = ti.Vector([x, y])
        temp_scale_col[i] = temperature_to_color(temp)

@ti.kernel
def update_visuals(time: ti.f32, t_star: ti.f32, r_star_solar: ti.f32, alpha: ti.f32, epsilon: ti.f32):
    r_star_au = r_star_solar * SOLAR_RADIUS_TO_AU
    p_star = pos_bodies[0]
    p_planet = pos_bodies[1]

    # Distance & surface temperature:
    diff = p_planet - p_star
    d_current = ti.sqrt(diff.dot(diff))
    planet_dist[None] = d_current

    t_curr = compute_planet_temp(d_current, t_star, r_star_au, alpha, epsilon)
    planet_temp[None] = t_curr

    # Update HZ boundaries:
    d_in = compute_hz_radius(T_INNER_HZ, t_star, r_star_au, alpha, epsilon)
    d_out = compute_hz_radius(T_OUTER_HZ, t_star, r_star_au, alpha, epsilon)
    hz_inner_dist[None] = d_in
    hz_outer_dist[None] = d_out

    # Planet display color:
    color_bodies[1] = temperature_to_color(t_curr)

    # Star display color:
    color_bodies[0] = star_blackbody_color(t_star)

    # Orbit Trail:
    h = trail_head[None]
    trail_pos[h] = p_planet
    trail_color[h] = color_bodies[1]
    trail_head[None] = (h + 1) % N_TRAIL
    if trail_len[None] < N_TRAIL:
        trail_len[None] += 1

    # Volumetric Mist:
    hz_w = d_out - d_in
    for i in range(N_HZ_GLOW):
        u = rand_hz[i][0]
        ang = rand_hz[i][1] + 0.05 * (time + float(i) * 0.001)
        r_hz = d_in + u * hz_w
        pos_hz[i] = p_star + ti.Vector([r_hz * ti.cos(ang), r_hz * ti.sin(ang)])

        mid = (u - 0.5) * 2.0
        alpha_glow = 1.0 - mid * mid
        t_sample = compute_planet_temp(r_hz, t_star, r_star_au, alpha, epsilon)
        base_c = temperature_to_color(t_sample)
        color_hz[i] = base_c * (0.4 + 0.6 * alpha_glow)

    # Stellar Corona:
    s_col = star_blackbody_color(t_star)
    s_rad = radius_bodies[0]
    for i in range(N_CORONA):
        th = phase_corona[i][0] + time * phase_corona[i][1]
        r_f = s_rad * (1.2 + 0.8 * phase_corona[i][2] + 0.3 * ti.sin(time * 3.0 + float(i)))
        pos_corona[i] = p_star + ti.Vector([r_f * ti.cos(th), r_f * ti.sin(th)])
        color_corona[i] = s_col * (0.6 + 0.4 * ti.sin(time * 5.0 + float(i) * 0.5))

    # Lagrange Points L1 - L5:
    u_vec = diff / (d_current + 1e-6)
    w_vec = ti.Vector([-u_vec[1], u_vec[0]])

    mu = mass_bodies[1] / (mass_bodies[0] + mass_bodies[1])
    gamma = ti.pow(mu / 3.0, 1.0 / 3.0)

    # L1: between star and planet
    lagrange_pos[0] = p_star + u_vec * (d_current * (1.0 - gamma))
    lagrange_color[0] = ti.Vector([1.0, 0.8, 0.2])

    # L2: behind planet
    lagrange_pos[1] = p_star + u_vec * (d_current * (1.0 + gamma))
    lagrange_color[1] = ti.Vector([1.0, 0.8, 0.2])

    # L3: opposite star
    lagrange_pos[2] = p_star - u_vec * (d_current * (1.0 + 5.0 / 12.0 * mu))
    lagrange_color[2] = ti.Vector([0.7, 0.7, 0.7])

    # L4: 60 deg ahead (Trojan)
    lagrange_pos[3] = p_star + d_current * (0.5 * u_vec + 0.8660254 * w_vec)
    lagrange_color[3] = ti.Vector([0.2, 1.0, 0.6])

    # L5: 60 deg behind (Greek)
    lagrange_pos[4] = p_star + d_current * (0.5 * u_vec - 0.8660254 * w_vec)
    lagrange_color[4] = ti.Vector([0.2, 1.0, 0.6])


# ==============================================================================
# 8. LINE VERTEX ASSEMBLY & COORDINATE PROJECTION KERNELS
# ==============================================================================
@ti.kernel
def assemble_lines(cam_cx: ti.f32, cam_cy: ti.f32, view_r: ti.f32, aspect: ti.f32,
                   show_grid: ti.i32, show_trail: ti.i32, show_hz_rings: ti.i32, grid_step: ti.f32):
    inv_w = 1.0 / (2.0 * view_r * aspect)
    inv_h = 1.0 / (2.0 * view_r)
    p_star = pos_bodies[0]
    p_planet = pos_bodies[1]

    d_in = hz_inner_dist[None]
    d_out = hz_outer_dist[None]

    # 1. Inner HZ Ring (Moist Greenhouse 340 K):
    if show_hz_rings == 1:
        for seg in range(N_RING_SEGS):
            th1 = float(seg) / float(N_RING_SEGS) * 2.0 * math.pi
            th2 = float(seg + 1) / float(N_RING_SEGS) * 2.0 * math.pi

            p1 = p_star + ti.Vector([d_in * ti.cos(th1), d_in * ti.sin(th1)])
            p2 = p_star + ti.Vector([d_in * ti.cos(th2), d_in * ti.sin(th2)])

            inner_ring_verts[seg * 2]     = ti.Vector([0.5 + (p1[0] - cam_cx) * inv_w, 0.5 + (p1[1] - cam_cy) * inv_h])
            inner_ring_verts[seg * 2 + 1] = ti.Vector([0.5 + (p2[0] - cam_cx) * inv_w, 0.5 + (p2[1] - cam_cy) * inv_h])

            inner_ring_color[seg * 2]     = ti.Vector([1.0, 0.7, 0.2])
            inner_ring_color[seg * 2 + 1] = ti.Vector([1.0, 0.7, 0.2])

        # 2. Outer HZ Ring (Freezing Point 273.15 K):
        for seg in range(N_RING_SEGS):
            th1 = float(seg) / float(N_RING_SEGS) * 2.0 * math.pi
            th2 = float(seg + 1) / float(N_RING_SEGS) * 2.0 * math.pi

            p1 = p_star + ti.Vector([d_out * ti.cos(th1), d_out * ti.sin(th1)])
            p2 = p_star + ti.Vector([d_out * ti.cos(th2), d_out * ti.sin(th2)])

            outer_ring_verts[seg * 2]     = ti.Vector([0.5 + (p1[0] - cam_cx) * inv_w, 0.5 + (p1[1] - cam_cy) * inv_h])
            outer_ring_verts[seg * 2 + 1] = ti.Vector([0.5 + (p2[0] - cam_cx) * inv_w, 0.5 + (p2[1] - cam_cy) * inv_h])

            outer_ring_color[seg * 2]     = ti.Vector([0.3, 0.8, 1.0])
            outer_ring_color[seg * 2 + 1] = ti.Vector([0.3, 0.8, 1.0])

    # 3. AU Reference Grid Rings:
    if show_grid == 1:
        for r_idx in range(N_GRID_RINGS):
            r_grid = grid_step * float(r_idx + 1)
            base_idx = r_idx * N_GRID_SEGS * 2
            g_col = ti.Vector([0.15, 0.20, 0.28])
            if ti.abs(r_grid - 1.0) < grid_step * 0.4:
                g_col = ti.Vector([0.28, 0.40, 0.55])

            for seg in range(N_GRID_SEGS):
                th1 = float(seg) / float(N_GRID_SEGS) * 2.0 * math.pi
                th2 = float(seg + 1) / float(N_GRID_SEGS) * 2.0 * math.pi

                p1 = p_star + ti.Vector([r_grid * ti.cos(th1), r_grid * ti.sin(th1)])
                p2 = p_star + ti.Vector([r_grid * ti.cos(th2), r_grid * ti.sin(th2)])

                idx = base_idx + seg * 2
                grid_verts[idx]     = ti.Vector([0.5 + (p1[0] - cam_cx) * inv_w, 0.5 + (p1[1] - cam_cy) * inv_h])
                grid_verts[idx + 1] = ti.Vector([0.5 + (p2[0] - cam_cx) * inv_w, 0.5 + (p2[1] - cam_cy) * inv_h])
                grid_color[idx]     = g_col
                grid_color[idx + 1] = g_col

    # 4. Orbit Trail Segments:
    if show_trail == 1:
        t_len = trail_len[None]
        t_head = trail_head[None]
        for seg in range(N_TRAIL - 1):
            if seg < t_len - 1:
                idx1 = (t_head - t_len + seg + N_TRAIL) % N_TRAIL
                idx2 = (idx1 + 1) % N_TRAIL

                p1 = trail_pos[idx1]
                p2 = trail_pos[idx2]

                fade = float(seg) / float(t_len)

                trail_line_verts[seg * 2]     = ti.Vector([0.5 + (p1[0] - cam_cx) * inv_w, 0.5 + (p1[1] - cam_cy) * inv_h])
                trail_line_verts[seg * 2 + 1] = ti.Vector([0.5 + (p2[0] - cam_cx) * inv_w, 0.5 + (p2[1] - cam_cy) * inv_h])

                trail_line_color[seg * 2]     = trail_color[idx1] * fade
                trail_line_color[seg * 2 + 1] = trail_color[idx2] * fade
            else:
                trail_line_verts[seg * 2]     = ti.Vector([0.0, 0.0])
                trail_line_verts[seg * 2 + 1] = ti.Vector([0.0, 0.0])
                trail_line_color[seg * 2]     = ti.Vector([0.0, 0.0, 0.0])
                trail_line_color[seg * 2 + 1] = ti.Vector([0.0, 0.0, 0.0])

    # 5. Diagnostic Vectors: Line-of-sight & Velocity vector:
    p1_s = ti.Vector([0.5 + (p_star[0] - cam_cx) * inv_w, 0.5 + (p_star[1] - cam_cy) * inv_h])
    p2_s = ti.Vector([0.5 + (p_planet[0] - cam_cx) * inv_w, 0.5 + (p_planet[1] - cam_cy) * inv_h])
    diag_lines_verts[0] = p1_s
    diag_lines_verts[1] = p2_s
    diag_lines_color[0] = ti.Vector([0.2, 0.3, 0.4])
    diag_lines_color[1] = ti.Vector([0.2, 0.3, 0.4])

    v_p = vel_bodies[1]
    v_arrow_end = p_planet + v_p * 0.04
    diag_lines_verts[2] = p2_s
    diag_lines_verts[3] = ti.Vector([0.5 + (v_arrow_end[0] - cam_cx) * inv_w, 0.5 + (v_arrow_end[1] - cam_cy) * inv_h])
    diag_lines_color[2] = ti.Vector([0.3, 1.0, 0.4])
    diag_lines_color[3] = ti.Vector([0.3, 1.0, 0.4])

    diag_lines_count[None] = 4


@ti.kernel
def project_points_to_screen(cam_cx: ti.f32, cam_cy: ti.f32, view_r: ti.f32, aspect: ti.f32):
    inv_w = 1.0 / (2.0 * view_r * aspect)
    inv_h = 1.0 / (2.0 * view_r)

    for i in range(N_DUST):
        p = pos_dust[i]
        screen_dust[i] = ti.Vector([0.5 + (p[0] - cam_cx) * inv_w, 0.5 + (p[1] - cam_cy) * inv_h])

    for i in range(N_HZ_GLOW):
        p = pos_hz[i]
        screen_hz[i] = ti.Vector([0.5 + (p[0] - cam_cx) * inv_w, 0.5 + (p[1] - cam_cy) * inv_h])

    for i in range(N_CORONA):
        p = pos_corona[i]
        screen_corona[i] = ti.Vector([0.5 + (p[0] - cam_cx) * inv_w, 0.5 + (p[1] - cam_cy) * inv_h])

    for b in range(2):
        p = pos_bodies[b]
        screen_bodies[b] = ti.Vector([0.5 + (p[0] - cam_cx) * inv_w, 0.5 + (p[1] - cam_cy) * inv_h])

    for lp in range(5):
        p = lagrange_pos[lp]
        screen_lagrange[lp] = ti.Vector([0.5 + (p[0] - cam_cx) * inv_w, 0.5 + (p[1] - cam_cy) * inv_h])


# ==============================================================================
# 9. INTERACTIVE CONTROLS & PRINT HELPER
# ==============================================================================
def print_welcome_banner():
    banner = r"""
================================================================================
          HABITABLE ZONE & PLANETARY ORBIT SIMULATOR (TAICHI GPU)
================================================================================
PHYSICAL MODEL:
  Habitable Zone Distance Formula:
    d = (R_star * T_star^2) / (2 * T_planet^2) * sqrt( 2*(1 - alpha) / (2 - epsilon) )
  where:
    * T_star   = Stellar Temperature in Kelvin (Sun = 5778 K)
    * R_star   = Stellar Radius in Solar Radii (Sun = 1.0 R_sun = 0.00465 AU)
    * T_planet = Surface Temperature in Kelvin (Liquid Water: 273.15 K to 340 K)
    * alpha    = Planetary Albedo (Earth ~ 0.306)
    * epsilon  = Atmospheric Emissivity / Greenhouse Effect (Earth ~ 0.77)

KEYBOARD & MOUSE CONTROLS:
  [SPACE]      : Pause / Resume simulation
  [R]          : Reset to Earth-Sun default baseline
  [C]          : Cycle camera center (Origin / Star / Planet)
  [G]          : Toggle AU distance reference grid
  [H]          : Toggle Habitable Zone volumetric mist
  [D]          : Toggle Circumstellar Dust & Asteroid Belt
  [T]          : Toggle Planetary Orbit Trail
  [L]          : Toggle Lagrange Point holographic reticles
  [+] / [-]    : Zoom In / Zoom Out
  [ [ ] / [ ] ]: Slow down / Speed up time step
  Left-Click Drag on canvas : Pan camera in AU space
  Right-Click on canvas     : Relocate planet to cursor and set circular orbit!

EMERGENT PHENOMENA VISIBLE:
  * Resonant clearing of dust feeding zone along the planet's orbit.
  * Stable Trojan asteroid swarms trapped around L4 and L5 Lagrange points!
  * Thermal color evolution: planet roasts at perihelion and freezes at aphelion!
  * Stellar wobble / reflex motion around system barycenter when planet mass is high.
================================================================================
"""
    print(banner)
    sys.stdout.flush()


# ==============================================================================
# 10. MAIN SIMULATION & INTERACTIVE RENDER LOOP
# ==============================================================================
def main():
    print_welcome_banner()

    # Simulation State Parameters (Default: Earth-Sun System):
    t_star = 5778.0         # Star temperature [K]
    r_star_solar = 1.0      # Star radius [R_sun]
    a_planet = 1.0          # Semi-major axis [AU]
    e_planet = 0.0167       # Eccentricity (Earth ~ 0.0167)
    m_planet_earth = 1.0    # Planet mass [M_earth]
    alpha = 0.306           # Bond albedo (Earth = 0.306)
    epsilon = 0.770         # Atmosphere emissivity / greenhouse (Earth = 0.770)

    # Simulation runtime controls:
    is_paused = False
    sim_speed = 1.0         # Time acceleration multiplier
    substeps_per_frame = 2 # Substeps of Velocity Verlet per display frame
    dt_base = 0.001       # Base integration substep (~3.0 hours per step)

    # Camera & Visualization Toggles:
    cam_cx = 0.0
    cam_cy = 0.0
    view_radius = 2.4       # View half-height in AU
    cam_mode = 1
    lock_star_position = True            # 0: Fixed Origin, 1: Star Centered, 2: Planet Centered

    show_grid = 1
    show_trail = 1
    show_dust = 1
    show_hz_glow = 1
    show_hz_rings = 1
    show_lagrange = 1

    # Mouse drag tracking:
    is_dragging = False
    last_mouse_x = 0.5
    last_mouse_y = 0.5

    # Initialize simulation fields:
    init_simulation(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)

    # Create Taichi GUI Window:
    window = ti.ui.Window("Habitable Zone & Planetary Orbit Simulation [Taichi GPU]", (WIN_W, WIN_H), vsync=False)
    canvas = window.get_canvas()
    gui = window.get_gui()

    sim_time = 0.0

    # Main Render & Physics Loop:
    while window.running:
        # ----------------------------------------------------------------------
        # A. USER INPUT & KEYBOARD EVENTS
        # ----------------------------------------------------------------------
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.SPACE:
                is_paused = not is_paused
            elif e.key == 'r' or e.key == 'R':
                t_star = 5778.0
                r_star_solar = 1.0
                a_planet = 1.0
                e_planet = 0.0167
                m_planet_earth = 1.0
                alpha = 0.306
                epsilon = 0.770
                cam_cx = 0.0
                cam_cy = 0.0
                view_radius = 2.4
                init_simulation(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)
            elif e.key == 'c' or e.key == 'C':
                cam_mode = (cam_mode + 1) % 3
            elif e.key == 'g' or e.key == 'G':
                show_grid = 1 - show_grid
            elif e.key == 't' or e.key == 'T':
                show_trail = 1 - show_trail
            elif e.key == 'd' or e.key == 'D':
                show_dust = 1 - show_dust
            elif e.key == 'h' or e.key == 'H':
                show_hz_glow = 1 - show_hz_glow
            elif e.key == 'l' or e.key == 'L':
                show_lagrange = 1 - show_lagrange
            elif e.key == '=' or e.key == '+':
                view_radius = max(0.4, view_radius * 0.85)
            elif e.key == '-' or e.key == '_':
                view_radius = min(8.0, view_radius * 1.15)
            elif e.key == ']':
                sim_speed = min(5.0, sim_speed * 1.3)
            elif e.key == '[':
                sim_speed = max(0.1, sim_speed / 1.3)

        # Mouse Drag Pan Interaction:
        mouse_x, mouse_y = window.get_cursor_pos()
        if window.is_pressed(ti.ui.LMB):
            if not is_dragging:
                is_dragging = True
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
            else:
                dx = mouse_x - last_mouse_x
                dy = mouse_y - last_mouse_y
                cam_cx -= dx * (2.0 * view_radius * ASPECT)
                cam_cy -= dy * (2.0 * view_radius)
                last_mouse_x = mouse_x
                last_mouse_y = mouse_y
                cam_mode = 0
                lock_star_position = False
        else:
            is_dragging = False

        # Right-Click to spawn or slingshot planet into cursor location:
        if window.is_pressed(ti.ui.RMB):
            w_x = cam_cx + (mouse_x - 0.5) * (2.0 * view_radius * ASPECT)
            w_y = cam_cy + (mouse_y - 0.5) * (2.0 * view_radius)
            new_r = math.sqrt(w_x * w_x + w_y * w_y)
            if new_r > 0.15:
                a_planet = float(new_r)
                e_planet = 0.0
                reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)

    
        # Camera centering mode:
        if lock_star_position:
            star_p = pos_bodies[0].to_numpy()
            cam_cx = float(star_p[0])
            cam_cy = float(star_p[1])
        elif cam_mode == 1:
            star_p = pos_bodies[0].to_numpy()
            cam_cx = float(star_p[0])
            cam_cy = float(star_p[1])
        elif cam_mode == 2:
            planet_p = pos_bodies[1].to_numpy()
            cam_cx = float(planet_p[0])
            cam_cy = float(planet_p[1])

        # ----------------------------------------------------------------------
        # B. SYMPLECTIC PHYSICS UPDATE (VELOCITY VERLET)
        # ----------------------------------------------------------------------
        if not is_paused:
            dt = dt_base * sim_speed
            for _ in range(substeps_per_frame):
                verlet_step_positions(dt)
                compute_accelerations(t_star, r_star_solar, alpha, epsilon)
                verlet_step_velocities(dt)

            sim_time += dt * substeps_per_frame

        # Update visuals, Habitable Zone boundaries, and history trail:
        update_dust_colors(
    t_star,
    r_star_solar,
    alpha,
    epsilon
)
        
        update_visuals(sim_time, t_star, r_star_solar, alpha, epsilon)
        update_temperature_scale()

        # Compute adaptive grid distance ring step based on current zoom:
        raw_step = view_radius / 4.0
        order = 10.0 ** math.floor(math.log10(max(1e-4, raw_step)))
        norm = raw_step / order
        if norm < 2.0:
            grid_step = 1.0 * order
        elif norm < 5.0:
            grid_step = 2.0 * order
        else:
            grid_step = 5.0 * order

        # Assemble lines and transform world AU points to canvas coordinates:
        assemble_lines(cam_cx, cam_cy, view_radius, ASPECT, show_grid, show_trail, show_hz_rings, float(grid_step))
        project_points_to_screen(cam_cx, cam_cy, view_radius, ASPECT)

        # ----------------------------------------------------------------------
        # C. MULTI-LAYER CINEMATIC RENDERING (CANVAS)
        # ----------------------------------------------------------------------
        canvas.set_background_color((0.015, 0.018, 0.025))

        #Temperature scale
        canvas.circles(temp_scale_pos, radius = 0.006, per_vertex_color=temp_scale_col)

        

        # Layer 1: Concentric AU Distance Reference Grid:
        if show_grid == 1:
            canvas.lines(grid_verts, width=0.0015, per_vertex_color=grid_color)

        # Layer 2: Habitable Zone Volumetric Radiant Mist:
        if show_hz_glow == 1:
            canvas.circles(screen_hz, radius=0.0028, per_vertex_color=color_hz)

        # Layer 3: Sharp Habitable Zone Boundary Rings:
        if show_hz_rings == 1:
            canvas.lines(inner_ring_verts, width=0.003, per_vertex_color=inner_ring_color)
            canvas.lines(outer_ring_verts, width=0.003, per_vertex_color=outer_ring_color)

        # Layer 4: Circumstellar Dust Disk & Trojan Asteroids:
        if show_dust == 1:
            canvas.circles(screen_dust, radius=0.0018, per_vertex_color=color_dust)

        # Layer 5: Planetary Orbit Trail (Color-coded by thermal history):
        if show_trail == 1:
            canvas.lines(trail_line_verts, width=0.0035, per_vertex_color=trail_line_color)

        # Layer 6: Diagnostic Line-of-sight & Velocity Vector:
        canvas.lines(diag_lines_verts, width=0.002, per_vertex_color=diag_lines_color)

        # Layer 7: Lagrange Points L1 - L5:
        if show_lagrange == 1:
            canvas.circles(screen_lagrange, radius=0.005, per_vertex_color=lagrange_color)

        # Layer 8: Stellar Corona Flares:
        canvas.circles(screen_corona, radius=0.0025, per_vertex_color=color_corona)

        # Layer 9: Central Star & Planet:
        star_screen_r = max(0.012, 0.016 * r_star_solar * (2.4 / view_radius))
        planet_screen_r = max(0.006, 0.008 * math.pow(m_planet_earth, 0.2) * (2.4 / view_radius))

        # Star Core & Atmosphere:
        canvas.circles(screen_bodies, radius=star_screen_r, per_vertex_color=color_bodies)
        
        star_halo_pos[0] = screen_bodies[0]
        star_halo_col[0] = color_bodies[0] * 0.45
        canvas.circles(star_halo_pos, radius=star_screen_r * 1.6, per_vertex_color=star_halo_col)

        # Planet Core & Atmosphere Halo:
        
        planet_pt[0] = screen_bodies[1]
        planet_col[0] = color_bodies[1]
        
        planet_halo_col[0] = color_bodies[1] * (0.3 + 0.5 * epsilon)
        canvas.circles(planet_pt, radius=planet_screen_r * 1.8, per_vertex_color=planet_halo_col)
        canvas.circles(planet_pt, radius=planet_screen_r, per_vertex_color=planet_col)

        # ----------------------------------------------------------------------
        # D. INTERACTIVE GUI CONTROLS & REAL-TIME PHYSICS DIAGNOSTICS
        # ----------------------------------------------------------------------
        curr_dist = float(planet_dist[None])
        curr_temp = float(planet_temp[None])
        curr_temp_c = curr_temp - 273.15
        d_in = float(hz_inner_dist[None])
        d_out = float(hz_outer_dist[None])
        hz_width = d_out - d_in

        lum_ratio = (r_star_solar ** 2) * ((t_star / 5778.0) ** 4)

        spectral_class = "G (Sun-like)"
        if t_star < 3700:
            spectral_class = "M (Red Dwarf)"
        elif t_star < 5200:
            spectral_class = "K (Orange Dwarf)"
        elif t_star < 6000:
            spectral_class = "G (Yellow Dwarf)"
        elif t_star < 7500:
            spectral_class = "F (Yellow-White)"
        elif t_star < 10000:
            spectral_class = "A (White Star)"
        else:
            spectral_class = "B (Blue-White Giant)"

        v_vec = vel_bodies[1].to_numpy()
        speed_au_yr = math.sqrt(v_vec[0]**2 + v_vec[1]**2)
        speed_km_s = speed_au_yr * AU_PER_YR_TO_KM_S

        if curr_temp > T_INNER_HZ:
            climate_status = "BOILING / MOIST GREENHOUSE (Ocean Vaporization)"
        elif curr_temp < T_OUTER_HZ:
            climate_status = "FROZEN SNOWBALL (Global Glaciation / Ice Sheets)"
        else:
            climate_status = "HABITABLE ZONE (Liquid Water Oceans Stable!)"

        gui.begin("Stellar & Planetary System Controls", 0.015, 0.02, 0.33, 0.96)

        gui.text("=== CENTRAL STAR PROPERTIES ===")
        t_star_new = gui.slider_float("Star Temp T0 (K)", t_star, 2200.0, 10000.0)
        r_star_new = gui.slider_float("Star Radius R (R_sun)", r_star_solar, 0.1, 3.0)

        gui.text(f"Luminosity: {lum_ratio:.3f} L_sun | Class: {spectral_class}")

        gui.text("")
        gui.text("=== PLANET ORBIT & ATMOSPHERE ===")
        a_new = gui.slider_float("Semi-major Axis a (AU)", a_planet, 0.2, 3.5)
        e_new = gui.slider_float("Orbit Eccentricity e", e_planet, 0.0, 0.85)
        m_new = gui.slider_float("Planet Mass (M_earth)", m_planet_earth, 0.1, 500.0)
        alpha_new = gui.slider_float("Bond Albedo alpha", alpha, 0.0, 0.95)
        eps_new = gui.slider_float("Emissivity epsilon", epsilon, 0.0, 0.99)

        param_changed = False
        orbit_changed = False

        if abs(t_star_new - t_star) > 1.0 or abs(r_star_new - r_star_solar) > 0.01:
            t_star = t_star_new
            r_star_solar = r_star_new
            param_changed = True

        if abs(a_new - a_planet) > 0.01 or abs(e_new - e_planet) > 0.005 or abs(m_new - m_planet_earth) > 0.1:
            a_planet = a_new
            e_planet = e_new
            m_planet_earth = m_new
            orbit_changed = True

        if abs(alpha_new - alpha) > 0.005 or abs(eps_new - epsilon) > 0.005:
            alpha = alpha_new
            epsilon = eps_new
            param_changed = True

        if orbit_changed:
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)

        gui.text("")
        gui.text("=== STELLAR EVOLUTION SEQUENCE ===")
        gui.text("Press sequentially to watch the HZ migrate:")
        
        for stage in evo_stages:
            if gui.button(stage["name"]):
                t_star = stage["t_star"]
                r_star_solar = stage["r_star"]
                a_planet = stage["a_planet"]
                e_planet = stage["e_planet"]
                m_planet_earth = stage["m_earth"]
                alpha = stage["alpha"]
                epsilon = stage["epsilon"]
                view_radius = stage["zoom"]
                reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)

        gui.text("")
        gui.text("=== ASTRONOMICAL PRESETS ===")
        if gui.button("Preset: Solar Earth (G2V Sun)"):
            t_star, r_star_solar = 5778.0, 1.0
            a_planet, e_planet, m_planet_earth = 1.0, 0.0167, 1.0
            alpha, epsilon = 0.306, 0.770
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)

        if gui.button("Preset: Mercury (Inner Thermal Extreme)"):
            t_star, r_star_solar = 5778.0, 1.0
            a_planet, e_planet, m_planet_earth = 0.387, 0.2056, 0.055
            alpha, epsilon = 0.088, 0.65
            view_radius = 1.2
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)
        if gui.button("Preset: Kepler-452b (Super-Earth)"):
            t_star, r_star_solar = 5757.0, 1.11
            a_planet, e_planet, m_planet_earth = 1.046, 0.035, 5.0
            alpha, epsilon = 0.30, 0.82
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)
        if gui.button("Preset: Wild Eccentric (Seasons of Fire & Ice)"):
            t_star, r_star_solar = 6000.0, 1.05
            a_planet, e_planet, m_planet_earth = 1.15, 0.62, 2.0
            alpha, epsilon = 0.306, 0.770
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)
        if gui.button("Preset: Venus (Runaway Greenhouse)"):
            t_star, r_star_solar = 5778.0, 1.0
            a_planet, e_planet, m_planet_earth = 0.723, 0.007, 0.815
            alpha, epsilon = 0.75, 0.99
            view_radius = 1.2
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)

        if gui.button("Preset: Mars (Frozen Desert)"):
            t_star, r_star_solar = 5778.0, 1.0
            a_planet, e_planet, m_planet_earth = 1.524, 0.093, 0.107
            alpha, epsilon = 0.25, 0.15
            view_radius = 2.0
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)
        if gui.button("Preset: Jovian Perturber (Resonant Gap & Trojans)"):
            t_star, r_star_solar = 5778.0, 1.0
            a_planet, e_planet, m_planet_earth = 1.30, 0.02, 318.0
            alpha, epsilon = 0.30, 0.77
            view_radius = 2.6
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)
        if gui.button("Preset: Kepler-22b (Super-Earth)"):
            t_star, r_star_solar = 5518.0, 0.979
            a_planet, e_planet, m_planet_earth = 0.849, 0.02, 9.1
            alpha, epsilon = 0.30, 0.77
            view_radius = 2.0
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)

        if gui.button("Preset: Hail Mary - Adrian (Tau Ceti e)"):
            t_star, r_star_solar = 5344.0, 0.79
            a_planet, e_planet, m_planet_earth = 0.55, 0.10, 4.0
            alpha, epsilon = 0.35, 0.65
            view_radius = 1.2
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)

        if gui.button("Preset: Hail Mary - Erid (40 Eridani A b)"):
            t_star, r_star_solar = 5000.0, 0.85
            a_planet, e_planet, m_planet_earth = 0.63, 0.05, 2.0
            alpha, epsilon = 0.30, 0.70
            view_radius = 1.2
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)

        if gui.button("Preset: Hot World (Extreme Greenhouse)"):
            t_star, r_star_solar = 5778.0, 1.0
            a_planet, e_planet, m_planet_earth = 0.55, 0.02, 1.0
            alpha, epsilon = 0.80, 0.20
            view_radius = 1.2
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)

        if gui.button("Preset: Frozen World (Low Emissivity)"):
            t_star, r_star_solar = 5778.0, 1.0
            a_planet, e_planet, m_planet_earth = 1.50, 0.05, 1.0
            alpha, epsilon = 0.30, 0.25
            view_radius = 2.0
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)

        if gui.button("Preset: Interstellar - Miller's Planet"):
            t_star, r_star_solar = 5778.0, 1.0
            a_planet, e_planet, m_planet_earth = 0.90, 0.05, 1.0
            alpha, epsilon = 0.10, 0.90
            view_radius = 1.5
            reset_planet_kepler(t_star, r_star_solar, a_planet, e_planet, m_planet_earth, alpha, epsilon)



        gui.text("")
        gui.text("=== HABITABLE ZONE MODEL FORMULA ===")
        gui.text("d = [R*T0^2 / (2*T^2)] * sqrt[ 2(1-alpha)/(2-epsilon) ]")
        gui.text("Where: T0=Star Temp, R=Star Radius (AU)")
        gui.text("       T=Planet Temp, alpha=Albedo, eps=Emissivity")

        gui.text("")
        gui.text("=== HABITABLE ZONE DIAGNOSTICS ===")
        gui.text(f"Inner Limit (340 K Moist GH) : {d_in:.4f} AU")
        gui.text(f"Outer Limit (273 K Freezing) : {d_out:.4f} AU")
        gui.text(f"Habitable Belt Width (Delta d): {hz_width:.4f} AU")
        gui.text(f"Planet Distance r(t)         : {curr_dist:.3f} AU ({curr_dist * AU_TO_KM / 1e6:.1f} M km)")
        gui.text(f"Orbital Velocity v(t)        : {speed_km_s:.2f} km/s")
        gui.text(f"Surface Temperature T_p(t)   : {curr_temp:.1f} K ({curr_temp_c:+.1f} deg C)")

        gui.text("")
        gui.text("=== TEMPERATURE COLOR SCALE ===")
        gui.text("Blue/Purple : < 200 K  (Very Cold)")
        gui.text("Cyan/Blue   : 200-273 K (Frozen)")
        gui.text("Green/Cyan  : 273-340 K (Habitable)")
        gui.text("Orange/Red  : 340-450 K (Greenhouse)")
        gui.text("Red         : > 450 K   (Very Hot)")

        gui.text("")
        gui.text("=== PLANETARY CLIMATE STATUS ===")
        gui.text(climate_status)

        temp_norm = max(0.0, min(1.0, (curr_temp - 150.0) / 350.0))
        gui.slider_float("Thermal Gauge", temp_norm, 0.0, 1.0)

        gui.text("")
        gui.text("=== SIMULATION & CAMERA CONTROLS ===")
        lock_star_position = bool(gui.checkbox("Lock Star Position", lock_star_position))
        sim_speed = gui.slider_float("Time Speed Multiplier", sim_speed, 0.1, 4.0)
        view_radius = gui.slider_float("Camera Zoom (AU Radius)", view_radius, 0.3, 5.0)

        cam_names = ["Origin (0,0)", "Track Star", "Track Planet"]
        gui.text(f"Camera Mode: {cam_names[cam_mode]}")
        if gui.button("Cycle Camera Center [C]"):
            cam_mode = (cam_mode + 1) % 3

        if gui.button("Reset Camera to Center"):
            cam_cx = 0.0
            cam_cy = 0.0
            view_radius = 2.4

        gui.text("")
        gui.text("=== VISUAL LAYERS TOGGLES ===")
        show_dust = int(gui.checkbox("Circumstellar Dust Disk [D]", bool(show_dust)))
        show_hz_glow = int(gui.checkbox("Habitable Zone Radiant Mist [H]", bool(show_hz_glow)))
        show_hz_rings = int(gui.checkbox("Habitable Zone Boundary Rings", bool(show_hz_rings)))
        show_trail = int(gui.checkbox("Thermal Orbit Trail [T]", bool(show_trail)))
        show_lagrange = int(gui.checkbox("Lagrange Points L1-L5 Reticles [L]", bool(show_lagrange)))
        show_grid = int(gui.checkbox("AU Reference Distance Grid [G]", bool(show_grid)))

        gui.end()

        # Render complete frame:
        window.show()

    window.destroy()


if __name__ == "__main__":
    main()