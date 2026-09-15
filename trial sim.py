import taichi as ti

# Initialize Taichi (automatically picks GPU or CPU)
ti.init()

# Simulation Parameters & Constants
num_planets = 1
G = 1.0  # Gravitational constant scaled for simulation
stellar_mass = 100.0
dt = 0.005  # Time step

# Taichi Fields for State Variables
x = ti.field(dtype=ti.f32, shape=(num_planets,))
y = ti.field(dtype=ti.f32, shape=(num_planets,))
vx = ti.field(dtype=ti.f32, shape=(num_planets,))
vy = ti.field(dtype=ti.f32, shape=(num_planets,))
ax = ti.field(dtype=ti.f32, shape=(num_planets,))
ay = ti.field(dtype=ti.f32, shape=(num_planets,))
current_distance = ti.field(dtype=ti.f32, shape=(num_planets,))


@ti.kernel
def update_orbit():
    # 1. Compute acceleration based on current positions
    for i in range(num_planets):
        r_sq = x[i] ** 2 + y[i] ** 2
        r = ti.sqrt(r_sq)
        ax[i] = -G * stellar_mass * x[i] / (r * r_sq)
        ay[i] = -G * stellar_mass * y[i] / (r * r_sq)

    # 2. Advance positions and velocities (Verlet step)
    for i in range(num_planets):
        # Position update
        x[i] += vx[i] * dt + 0.5 * ax[i] * dt**2
        y[i] += vy[i] * dt + 0.5 * ay[i] * dt**2

        # Save old acceleration
        old_ax = ax[i]
        old_ay = ay[i]

        # Recompute acceleration at new position
        r_sq = x[i] ** 2 + y[i] ** 2
        r = ti.sqrt(r_sq)
        ax[i] = -G * stellar_mass * x[i] / (r * r_sq)
        ay[i] = -G * stellar_mass * y[i] / (r * r_sq)

        # Velocity update
        vx[i] += 0.5 * (old_ax + ax[i]) * dt
        vy[i] += 0.5 * (old_ay + ay[i]) * dt

        # Track active distance for UI/Habitable Zone triggers
        current_distance[i] = r


def init_orbit():
    # Set orbital parameters: semi-major axis (a) and eccentricity (e)
    a = 1.5
    e = 0.01  # Noticeable ellipse

    # Initialize planet at periapsis on the x-axis
    x[0] = a * (1.0 - e)
    y[0] = 0.0
    vx[0] = 0.0
    vy[0] = (
        G * stellar_mass / a * ((1.0 + e) / (1.0 - e))
    ) ** 0.5  # Vis-viva velocity


def main():
    init_orbit()
    gui = ti.GUI("Simathon - Star & Planet Orbit", res=800, background_color=0x0B0F19)

    while gui.running:
        # Run multiple sub-steps per frame to make the animation smooth
        for _ in range(10):
            update_orbit()

        # Clear screen and draw background grid/elements
        gui.clear(0x0B0F19)

        # Draw Star at the center (normalized screen coordinates: center is 0.5, 0.5)
        gui.circle((0.5, 0.5), color=0xFFD700, radius=18)

        # Get current world position of the planet
        px = x.to_numpy()
        py = y.to_numpy()

        # Map world coordinates to screen space (scaling factor of 8.0 to fit view bounds)
        screen_x = 0.5 + px[0] / 8.0
        screen_y = 0.5 + py[0] / 8.0

        # Draw Planet
        gui.circle((screen_x, screen_y), color=0x4EA8DE, radius=7)

        gui.show()


if __name__ == "__main__":
    main()