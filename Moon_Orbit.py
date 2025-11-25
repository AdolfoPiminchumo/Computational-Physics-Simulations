import numpy as np
import scipy.integrate as si
import matplotlib.pyplot as plt
import time
from matplotlib.ticker import FormatStrFormatter

# =============== CONSTANTS DEFINITION =====================
# Physical constants for Earth-Moon system
G = 6.67430e-11       # Gravitational constant (m^3 kg^-1 s^-2)
M_e = 5.972e24    # Mass of Earth (kg)
M_m = 7.348e22     # Mass of Moon (kg)
r_m = 3.844e8    # Average Earth-Moon distance (m)
AU = 1.496e11         # Astronomical unit (m)
earth_radius = 6.371e6  # Earth's radius (m)


# =================== FUNCTION PHYSICS MODELS ================

def MoonOrbitAboutEarth(time, state, M_e, M_m):
    """
    Compute derivatives for Moon's orbital motion around Earth.
    
    This function defines the system of differential equations for a two-body
    problem where the Moon orbits the Earth under gravitational influence.
    
    Parameters
    ----------
    time : float
        Independent variable (time in seconds). Not used since
        the gravitational forces are time-independent.
    state : tuple of floats
        Current state vector containing:
        - x_m : float, Moon's x-position (m)
        - y_m : float, Moon's y-position (m)  
        - vx_m : float, Moon's x-velocity (m/s)
        - vy_m : float, Moon's y-velocity (m/s)
    mass_earth : float
        Mass of Earth (kg)
    mass_moon : float
        Mass of Moon (kg) - included for consistency with general two-body form
        
    Returns
    -------
    tuple of floats
        Derivatives of the state vector:
        - dx/dt : x-velocity (m/s)
        - dy/dt : y-velocity (m/s) 
        - dvx/dt : x-acceleration (m/s^2)
        - dvy/dt : y-acceleration (m/s^2)
        
    Notes
    -----
    This assumes Earth is fixed at the origin (relatively good approximation when M_earth >> M_moon).
    
    
    """
    # Current state
    x_m, y_m, vm_x, vm_y = state

    # ===================== VELOCITY DERIVATIVES =====================
    f1 = vm_x  # dx_m/dt - Moon x-velocity (m/s)
    f2 = vm_y  # dy_m/dt - Moon y-velocity (m/s)
    
    # ===================== DISTANCE CALCULATIONS =====================
    # Calculate distances between all bodies
    r_m = np.sqrt(x_m**2 + y_m**2)        # Earth-Moon distance (m)

    # ===================== ACCELERATION DERIVATIVES =====================
    f3 = - (M_e*G*x_m)/r_m**3  # dvm_x/dt - Moon x-acceleration (m/s^2)
    f4 = - (M_e*G*y_m)/r_m**3  # dvm_y/dt - Moon x-acceleration (m/s^2)

    return (f1,f2,f3,f4)


def MoonEarthProbe(time, state, M_e, M_m):
    """
    Function to compute the derivatives for the Earth-Moon-Probe three-body system.
    
    This function defines the system of differential equations for a restricted
    three-body problem where both the Moon and a probe move under the gravitational
    influence of Earth, and the probe is additionally influenced by the Moon.
    
    Parameters 
    ----------
    time : float
        Independent variable (time in seconds). Not used explicitly since
        the gravitational forces are time-independent.
    state : tuple of floats
        Current state vector containing:
        - x_m, y_m : Moon position coordinates (m)
        - x_p, y_p : Probe position coordinates (m)  
        - vm_x, vm_y : Moon velocity components (m/s)
        - vp_x, vp_y : Probe velocity components (m/s)
    M_e : float
        Mass of Earth (kg)
    M_m : float
        Mass of Moon (kg)
        
    Returns
    -------
    tuple of floats
        Derivatives of the state vector in order:
        - dx_m/dt, dy_m/dt : Moon velocity components
        - dx_p/dt, dy_p/dt : Probe velocity components  
        - dvm_x/dt, dvm_y/dt : Moon acceleration components
        - dvp_x/dt, dvp_y/dt : Probe acceleration components
    """
    
    # State vector
    x_m, y_m, x_p, y_p, vm_x, vm_y, vp_x, vp_y = state

    # ===================== VELOCITY DERIVATIVES =====================
    # These are straightforward: derivative of position = velocity
    f1 = vm_x  # dx_m/dt - Moon x-velocity (m/s)
    f2 = vm_y  # dy_m/dt - Moon y-velocity (m/s)
    f3 = vp_x  # dx_p/dt - Probe x-velocity (m/s)  
    f4 = vp_y  # dy_p/dt - Probe y-velocity (m/s)

    # ===================== DISTANCE CALCULATIONS =====================
    # Calculate distances between all bodies
    r_m = np.sqrt(x_m**2 + y_m**2)        # Earth-Moon distance (m)
    r_p = np.sqrt(x_p**2 + y_p**2)        # Earth-Probe distance (m)
    r_pm = np.sqrt((x_p - x_m)**2 + (y_p - y_m)**2)  # Moon-Probe distance (m)

    # ===================== ACCELERATION DERIVATIVES =====================
    # Moon acceleration (only influenced by Earth since probe mass is negligible)
    f5 = - (M_e * G * x_m) / r_m**3  # dvm_x/dt - Moon x-acceleration (m/s^2)
    f6 = - (M_e * G * y_m) / r_m**3  # dvm_y/dt - Moon y-acceleration (m/s^2)

    # Probe acceleration (influenced by both Earth and Moon)
    # Earth's gravitational influence on probe
    earth_probe_x = - (M_e * G * x_p) / r_p**3
    earth_probe_y = - (M_e * G * y_p) / r_p**3
    
    # Moon's gravitational influence on probe  
    moon_probe_x = - (M_m * G * (x_p - x_m)) / r_pm**3
    moon_probe_y = - (M_m * G * (y_p - y_m)) / r_pm**3
    
    # Total probe acceleration = Earth influence + Moon influence
    f7 = earth_probe_x + moon_probe_x  # dvp_x/dt - Probe x-acceleration (m/s^2)
    f8 = earth_probe_y + moon_probe_y  # dvp_y/dt - Probe y-acceleration (m/s^2)

    return (f1, f2, f3, f4, f5, f6, f7, f8)


# =============== MAIN SIMULATION FUNCTIONS =======================
def simulation_earth_moon(n_orbit, vel_frac):

    """
    Simulate the Moon's orbit around Earth for different numerical tolerances.
    
    Orbital simulation using solve_ivp with varying tolerance levels
    and plots both the orbital trajectories and energy conservation.
    
    Parameters
    ----------
    n_orbit : float
        Number of orbital periods to simulate
    vel_frac : float
        Fraction of circular orbital velocity to use as initial y-velocity
        (1.0 = circular orbit, <1.0 = elliptical orbit, >1.0 = hyperbolic trajectory)
    
    """
    # ===================== INITIAL CONDITIONS =====================
    # Moon initial state: starting on x-axis with vertical velocity
    x_m_0 = r_m                    # Initial x-position (m) - at average Earth-Moon distance
    y_m_0 = 0.0                    # Initial y-position (m) - starting on x-axis
    vm_x_0 = 0.0                   # Initial x-velocity (m/s) - no horizontal component
    vm_y_0 = np.sqrt(G * M_e / r_m) * vel_frac  # Initial y-velocity (m/s) - circular orbit fraction
    
    initial_state = (x_m_0, y_m_0, vm_x_0, vm_y_0)


    # ===================== TIME PARAMETERS =====================
    
    if abs(vel_frac - 1.0) < 1e-10:
        # Circular orbit case
        orbital_period = 2 * np.pi * r_m / vm_y_0
    else:
        # Elliptical orbit - use energy to find semi-major axis Kepler's law
        initial_energy = -G * M_e * M_m / r_m + 0.5 * M_m * vm_y_0**2
        semi_major_axis = -G * M_e * M_m / (2 * initial_energy)
        orbital_period = 2 * np.pi * np.sqrt(semi_major_axis**3 / (G * M_e))
    
    t_min = 0.0                                # Start time (s)
    t_max = n_orbit * orbital_period           # End time (s)
    numpoints = 1001                           # Number of evaluation points
    t_eval = np.linspace(t_min, t_max, numpoints)
    
    # ===================== SOLVER TOLERANCES =====================
    tolerance = [1e-6, 1e-3, 1e-2]  # rtol and atol tolerance values to test


    # ===================== SETUP PLOTS =====================
    plt.figure(figsize=(10, 8))
    ax = plt.axes()
    ax.set_aspect(1)
    ax.set_xlabel("x coordinate (AU)",  fontsize = 16)
    ax.set_ylabel("y coordinate (AU)", fontsize = 16)
    ax.set_title(rf"Moon Orbit Around Earth - {n_orbit} Orbits, $v_{{\mathrm{{fraction}}}}$ = {vel_frac}", usetex=True,  fontsize = 18)

    # Add Earth as a circle at the origin 
    earth_circle = plt.Circle((0, 0), earth_radius/AU, color='blue', alpha=0.7, label='Earth')
    ax.add_patch(earth_circle)

    # Figure for energy conservation
    plt.figure(figsize=(10, 6))
    ax2 = plt.axes()
    ax2.set_xlabel("Time (s)",  fontsize = 16)
    ax2.set_ylabel(r"Normalized Energy $(E/E_0)$", fontsize = 16)
    ax2.set_title("Energy Conservation for Different Tolerances", fontsize = 18)
    ax2.grid(True)


    # figure for angular momentum
    plt.figure(figsize=(10, 6))
    ax3 = plt.axes()
    #plt.plot(results.t/3600, scaled_momentum, 'purple', linewidth=2)
    ax3.set_xlabel("Time (hours)", fontsize = 18)
    ax3.set_ylabel("Angular Momentum ($\\times 10^{34}$ kg·m^2/s)", fontsize = 16)
    ax3.set_title("Angular Momentum Conservation", fontsize = 16)
    ax3.grid(True)
    
    # ===================== RUN SIMULATIONS =====================
    print(f"Running simulation for {n_orbit} orbits with velocity fraction {vel_frac}")
    print("-" * 60)

    for i, tol in enumerate(tolerance):

        print(f"Running simulation with tolerance: {tol}")
        # Start the timer
        start_time = time.time()

        # ------------- ORBIT PART ------------------
        results = si.solve_ivp(MoonOrbitAboutEarth, 
                            (t_min, t_max), 
                            initial_state,
                            method='RK45',
                            t_eval=t_eval,
                            args=(M_e, M_m), 
                            rtol=tol,
                            atol=tol
                            )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        
        # Check if simulation was successful

        if not results.success:
            print(f"Simulation failed with tolerance {tol}: {results.message}")
            continue  # skip next simulation if current fails
            
        print(f"  Success: {results.y.shape[1]} points computed")
        # Print the time
        print(f"  Simulation completed in {elapsed_time:.5f} seconds")

        # ===================== ENERGY CALCULATION ======================
        accumulated_energy = []
        for j in range(results.y.shape[1]):
            # Extract state at each time point
            state_at_t = [
                results.y[0][j],  # x-position
                results.y[1][j],  # y-position  
                results.y[2][j],  # x-velocity
                results.y[3][j]   # y-velocity
            ]
            total_energy = energy(state_at_t, 1, M_e, M_m)
            accumulated_energy.append(total_energy)
        
        # Normalize energy relative to initial energy
        initial_energy = accumulated_energy[0]
        normalized_energy = np.array(accumulated_energy) / initial_energy   

        # Calculate energy conservation error
        energy_error = np.max(np.abs(normalized_energy - 1.0))
        print(f"  Maximum energy error: {energy_error:.2e}")

        # ================= ANGULAR MOMENTUM ==================
        accumulated_angular_momentum = []
        for j in range(results.y.shape[1]):
            state_at_t = [
                results.y[0][j],  # x-position
                results.y[1][j],  # y-position  
                results.y[2][j],  # x-velocity
                results.y[3][j]   # y-velocity
            ]        
            total_momentum = AngularMomentum(state_at_t, 1, M_e, M_m)
            accumulated_angular_momentum.append(total_momentum)

        # Normalize angular momentum relative to initial momentum
        initial_momentum = accumulated_angular_momentum[0]
        normalized_momentum = np.array(accumulated_angular_momentum) / initial_momentum   

        # Calculate angular momenum conservation error
        momentum_error = np.max(np.abs(normalized_momentum - 1.0))
        print(f"  Maximum angular momentum error: {momentum_error:.2e}")
        print(f"  Angular momentum of the Moon: {accumulated_angular_momentum[-1]:.2e} Kg-m^2 / s \n")


        scaled_momentum = np.array(accumulated_angular_momentum)/1e34 

        # ===================== PLOTTING =====================
        # Plot orbital trajectory
        ax.plot(results.y[0]/AU, results.y[1]/AU, 
                label=f'Tol = {tol}', linewidth=1.5)
        
        # Plot energy conservation
        ax2.plot(results.t, normalized_energy,    # using solver time to avoid crashes in the code in case of divergence in the simulation
                label=f'Tol = {tol}', linewidth=1.5)
        
        # Plot angular momentum
        ax3.plot(results.t/3600, scaled_momentum, 
                 label=f'Tol = {tol}', linewidth=1.5)


     # ===================== FINALIZE PLOTS =====================
    # Finalize orbit plot
    ax.legend(loc='upper right', fontsize='small')
    ax.grid(True)

    # Finalize energy plot 
    ax2.legend(loc='upper right', fontsize='small')

    # Finalize angular momentum plot 
    ax3.legend(loc='upper right', fontsize='medium')

    plt.show()
    print("Simulation completed successfully!")

    return results


def simulation_earth_moon_probe(n_orbit, mass_earth_frac, vel_frac, pm_distance):

    """
    Simulate the Earth-Moon-Probe three-body system.
    
    Parameters
    ----------
    n_orbit : float
        Number of orbital periods to simulate
    mass_earth_frac : float
        Fraction of Earth's mass to use (1.0 = actual Earth mass)
    vel_frac : float
        Fraction of circular orbital velocity for Moon's initial velocity
    pm_distance : float
        Initial distance between probe and Moon (km)
        
    Notes
    -----
    State vector contains: (x_m, y_m, x_p, y_p, vm_x, vm_y, vp_x, vp_y)
    where:
    - x_m, y_m: Moon position coordinates (m)
    - x_p, y_p: Probe position coordinates (m)
    - vm_x, vm_y: Moon velocity components (m/s)
    - vp_x, vp_y: Probe velocity components (m/s)
    """

    # ===================== SYSTEM PARAMETERS =====================
    M_e = 5.972e24 * mass_earth_frac # Mass of Earth (kg)

    print("Simulation Parameters:")
    print(f"  Earth mass: {M_e:.2e} kg ({mass_earth_frac:.2f} × actual)")
    print(f"  Moon velocity fraction: {vel_frac:.2f}")
    print(f"  Initial probe-Moon distance: {pm_distance}  km")

    pm_distance = pm_distance * 1e3  # Convert to meters

   # ================== INITIAL CONDITIONS =====================
    # Moon initial conditions (starting on x-axis)
    x_m_0 = r_m                                  # Moon x-position (m)
    y_m_0 = 0.0                                 # Moon y-position (m)
    vm_x_0 = 0.0                                # Moon x-velocity (m/s)
    vm_y_0 = np.sqrt(G * M_e / r_m) * vel_frac  # Moon y-velocity (m/s)

    # Probe initial conditions (ahead of Moon in orbit)
    x_p_0 = r_m + pm_distance           # Probe x-position (m)
    y_p_0 = 0.0                                 # Probe y-position (m)
    vp_x_0 = 0.0                                # Probe x-velocity (m/s)
    vp_y_0 = vm_y_0 * 1.1                       # Probe y-velocity (m/s) - guess

    initial_state = (x_m_0, y_m_0, x_p_0, y_p_0, vm_x_0, vm_y_0, vp_x_0, vp_y_0)

    # ===================== TIME PARAMETERS =====================    

    if abs(vel_frac - 1.0) < 1e-5:
        # Circular orbit case
        orbital_period = 2 * np.pi * r_m / vm_y_0
    else:
        # Elliptical orbit - use energy to find semi-major axis
        initial_energy = -G * M_e * M_m / r_m + 0.5 * M_m * vm_y_0**2
        semi_major_axis = -G * M_e * M_m / (2 * initial_energy)
        orbital_period = 2 * np.pi * np.sqrt(semi_major_axis**3 / (G * M_e))
    
    t_min = 0.0                                 # Start time (s)
    t_max = n_orbit * orbital_period            # End time (s)
    numpoints = 1001                            # Number of evaluation points
    t_eval = np.linspace(t_min, t_max, numpoints)

    print(f"  Orbital period of the moon is: {orbital_period/86400:.2f} days \n")


    # ===================== RUN SIMULATIONS =====================
    tolerance = 1e-6
    print(f"Running simulation with tolerance: {tolerance}")

    # Start the timer
    start_time = time.time()

    results = si.solve_ivp(
        MoonEarthProbe,
        (t_min, t_max),
        initial_state,
        method='RK45',
        t_eval=t_eval,
        args=(M_e, M_m),
        rtol=tolerance,
        atol = tolerance
    )

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Check simulation success
    if not results.success:
        print(f"WARNING: Simulation failed: {results.message}")
        return
    
    print(f"   Simulation completed: {results.y.shape[1]} points computed")
    # Print the time
    print(f"   Simulation completed in {elapsed_time:.5f} seconds")

    # ===================== EXTRACT RESULTS =====================
    x_m, y_m = results.y[0], results.y[1]      # Moon trajectory
    x_p, y_p = results.y[2], results.y[3]      # Probe trajectory
    
    # Calculate probe-Moon distance over time
    p_m = np.sqrt((x_p - x_m)**2 + (y_p - y_m)**2)

    # ===================== ENERGY ANALYSIS =====================

    accumulated_energy = []
    for j in range(results.y.shape[1]):
        state_at_t = [x_m[j], y_m[j], x_p[j], y_p[j], results.y[4][j], 
                      results.y[5][j], results.y[6][j], results.y[7][j]]        
        total_energy = energy(state_at_t, 2, M_e, M_m)
        accumulated_energy.append(total_energy)
    

    # Normalize energy relative to initial energy
    normalized_energy = np.array(accumulated_energy) / accumulated_energy[0]
    energy_error = np.max(np.abs(normalized_energy - 1.0))
    print(f"   Maximum energy conservation error: {energy_error:.2e}")


    # ================== ANGULAR MOMENTUM ====================

    accumulated_angular_momentum = []
    for j in range(results.y.shape[1]):
        state_at_t = [x_m[j], y_m[j], x_p[j], y_p[j], results.y[4][j], 
                      results.y[5][j], results.y[6][j], results.y[7][j]]        
        total_momentum = AngularMomentum(state_at_t, 2, M_e, M_m)
        accumulated_angular_momentum.append(total_momentum)

    # Normalize momentum
    normalized_momentum = np.array(accumulated_angular_momentum) / accumulated_angular_momentum[0]
    momentum_error = np.max(np.abs(normalized_momentum - 1.0))
    print(f"   Maximum angular momentum error: {momentum_error:.2e}")

    scaled_momentum = np.array(accumulated_angular_momentum)/1e34

    # ====================== PLOTS =========================
    plt.figure(figsize=(10, 8))
    
    # Earth as a circle
    earth_radius = 6.371e6  # meters
    earth_circle = plt.Circle((0, 0), earth_radius / AU, color='blue', alpha=0.7, label='Earth')
    plt.gca().add_patch(earth_circle)

    # Moon and probe trajectories 
    plt.plot(x_m / AU, y_m / AU, 'blue', label='Moon orbit', alpha=0.7)
    plt.plot(x_p / AU, y_p / AU, 'red', label='Probe trajectory', alpha=0.7)

    # Labels and title
    plt.xlabel("x coordinate (AU)", fontsize = 16)
    plt.ylabel("y coordinate (AU)", fontsize = 16)
    plt.title(f"Earth-Moon-Probe System\n"
              f"Earth mass fraction: {mass_earth_frac:.1f}, Moon velocity fraction from a circular orbit: {vel_frac:.1f}, "
              f"Initial probe-Moon dist: {pm_distance/1000} km", fontsize = 18)
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.grid(True)


    plt.show()

    # --- Second plot: Probe-Moon distance vs time ---
    plt.figure(figsize=(10, 6))
    plt.yscale('log')
    plt.plot(results.t / 3600, p_m / 1000, 'green', linewidth=2)  # Convert time to hours, distance to km
    plt.xlabel("Time (hours)", fontsize = 16)
    plt.ylabel("Probe-Moon Distance (km)", fontsize = 16)
    plt.title("Probe-Moon Distance vs Time", fontsize = 18)
    plt.grid(True, alpha=0.3)
    plt.show()

    # ---- plot of energy
    plt.figure(figsize=(12, 6))
    plt.plot(results.t/3600, normalized_energy, 'purple', linewidth=2)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f')) # Formats the number to 9 decimal places

    plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect conservation')
    plt.xlabel("Time (hours)",  fontsize = 16)
    plt.ylabel("Normalized Energy (E/E₀)", fontsize = 16)
    plt.title(f"Energy Conservation (max error: {energy_error:.2e})", fontsize = 18)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # --- plot of momentum 

    plt.figure(figsize=(12, 6))
    plt.plot(results.t/3600, scaled_momentum, 'purple', linewidth=2)

    
    # Fix decimal places in the plot
    ax = plt.gca()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.4f')) # Formats the number to 4

    
# --- FIX ENDS HERE ---
    
    # Calculate a small padding for the y-axis (e.g., 10% of the data range)
    momentum_range = max(scaled_momentum) - min(scaled_momentum)
    y_padding_momentum = momentum_range *2
    # Apply the limits
    plt.ylim(min(scaled_momentum) - y_padding_momentum, max(scaled_momentum) + y_padding_momentum)
   

    plt.axhline(y = scaled_momentum[0], color='red', linestyle='--', alpha=0.7, label='Perfect conservation')
    plt.xlabel("Time (hours)", fontsize=16)
    plt.ylabel("Angular Momentum ($\\times 10^{34}$ kg·m^2/s)", fontsize=16)
    plt.title("Angular Momentum Conservation",  fontsize = 18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# =================== EXTRA FUNCTIONS ================
def AngularMomentum(state, system, M_e, M_m):
    """
    Calculate angular momentum conservation
    
    """
    if system == 1:
        x_m, y_m, vx_m, vy_m = state
        # L = r × p
        return M_m * (x_m * vy_m - y_m * vx_m)
    else:
        x_m, y_m, x_p, y_p, vx_m, vy_m, vx_p, vy_p = state
        M_PROBE = 1000
        L_moon = M_m * (x_m * vy_m - y_m * vx_m)
        L_probe = M_PROBE * (x_p * vy_p - y_p * vx_p)
        return L_moon + L_probe


def energy(state, system, M_e, M_m):
    """
    Calculate the total angular momentum of the Earth-Moon system with optional probe.

    For system 1: Only considers the Moon orbiting Earth (two-body problem).
    For system 2: Considers both Moon and probe orbiting Earth (three-body problem).
    The origin (0,0) is assumed to be the location of the Earth.

    Parameters
    ----------
    state : tuple of floats
        State vector containing positions and velocities:
        - System 1: [x_m, y_m, vx_m, vy_m]
        - System 2: [x_m, y_m, x_p, y_p, vx_m, vy_m, vx_p, vy_p]
    system : int
        System identifier: 1 for Moon-only, 2 for Moon+Probe.
    M_e : float
        Mass of Earth (kg). (Currently unused in this function).
    M_m : float
        Mass of Moon (kg).

    Returns
    -------
    float
        Total scalar angular momentum of the system (kg·m^2/s).

    Notes
    -----
    - Probe mass is assumed to be 1000 kg in system 2.
    - Total angular momentum should be conserved in the absence of external torques.
    """
    
    # ===================== SYSTEM 1: MOON ONLY =====================
    if system == 1:
        # Unpack state for Moon-only system
        x_m, y_m, vm_x, vm_y = state
        
        # Calculate distance and speed
        r = np.sqrt(x_m**2 + y_m**2)           # Earth-Moon distance (m)
        v = np.sqrt(vm_x**2 + vm_y**2)         # Moon speed (m/s)
        
        # Energy components for Moon
        KE = 0.5 * M_m * v**2                  # Kinetic energy (J)
        PE = -G * M_e * M_m / r                # Potential energy Earth-Moon (J)
        
        return KE + PE

     # ===================== SYSTEM 2: MOON + PROBE =====================
    else:
        # Unpack state for Moon+Probe system
        x_m, y_m, x_p, y_p, vx_m, vy_m, vx_p, vy_p = state
        
        # Constants
        M_PROBE = 1000  # Probe mass (kg)
        
        # ============ MOON ENERGY COMPONENTS ============
        r_em = np.sqrt(x_m**2 + y_m**2)        # Earth-Moon distance (m)
        v_m = np.sqrt(vx_m**2 + vy_m**2)       # Moon speed (m/s)
        KE_m = 0.5 * M_m * v_m**2              # Moon kinetic energy (J)
        PE_m = -G * M_e * M_m / r_em           # Moon potential energy Earth only (J)
        
        # ============ PROBE ENERGY COMPONENTS ============
        r_ep = np.sqrt(x_p**2 + y_p**2)        # Earth-Probe distance (m)
        r_mp = np.sqrt((x_p - x_m)**2 + (y_p - y_m)**2)  # Moon-Probe distance (m)
        v_p = np.sqrt(vx_p**2 + vy_p**2)       # Probe speed (m/s)
        
        # Probe kinetic energy
        KE_p = 0.5 * M_PROBE * v_p**2 
        
        # Probe potential energy (from both Earth and Moon)
        PE_earth_probe = -G * M_e * M_PROBE / r_ep 
        PE_moon_probe = -G * M_m * M_PROBE / r_mp
        PE_p = PE_earth_probe + PE_moon_probe
        
        # ============ TOTAL SYSTEM ENERGY ============
        total_energy = KE_m + PE_m + KE_p + PE_p
        return total_energy
   

# ======================== USER's INTERFACE ================================


MyInput = '1'
while MyInput != 'q':
    
    
    print("\n" + "="*60)
    print("        EARTH-MOON ORBITAL SIMULATION PROGRAM")
    print("="*60)
    print("This program simulates orbital dynamics in the Earth-Moon system.")
    print("Choose from the following options:")
    print("  1 - Moon orbit around Earth (Two-body problem)")
    print("  2 - Earth-Moon-Probe system (Three-body problem)")
    print("  q - Quit the program")
    print("="*60)
    MyInput = input('Enter a choice, "1", "2" or "q" to quit: ')

    print('You entered the choice: ',MyInput)
    if MyInput == '1':
        print("\n" + "-"*50)
        print("PART 1: MOON ORBIT AROUND EARTH")
        print("-"*50)
        
        # put your code for part (1) here
        number_orbits = input(
        "Enter number of orbital periods to simulate:\n"
        "• 0.5 = half orbit, 1 = full orbit, 2 = two orbits\n"
        "• Recommended: 1-5 orbits\n"
        "Your choice: "
        )
        
        # Avoid showing errors to user

        try:
            number_orbits = float(number_orbits)  # Try to convert to float ot int
        except ValueError:
            print('Wrong format, please type your value again')
    
            while True:
                try:
                    number_orbits = float(input("WRONG VALUE - Enter number of orbital periods to simulate: "))
                    break
                except ValueError:
                    print("Still not a valid number. Try again...")


        velocity = input(
        "Enter velocity as fraction of circular orbital speed:\n"
        "• 1.0 = perfect circular orbit\n" 
        "• <1.0 = elliptical orbit (e.g., 0.8)\n"
        "• >1.0 = may escape orbit (e.g., 1.2)\n"
        "• Recommended: 0.8-1.2\n"
        "Your choice: "
        )


        # Avoid showing errors to user

        try:
            velocity = float(velocity)  # Try to convert to float ot int
        except ValueError:
            print('Wrong format, please type your value again')
    
            while True:
                try:
                    velocity = float(input("WRONG VALUE - Enter velocity as fraction of circular orbital speed: "))
                    break
                except ValueError:
                    print("Still not a valid number. Try again...")


        print(f"\nStarting simulation with:")
        print(f" - Orbits: {number_orbits}")
        print(f" - Velocity fraction: {velocity}")
        print("...")
        simulation_earth_moon(number_orbits, velocity)
    
    elif MyInput == '2':
        print("\n" + "-"*50)
        print("PART 2: EARTH-MOON-PROBE SYSTEM")
        print("-"*50)

        number_orbits = input(
        "Enter number of orbital periods to simulate:\n"
        "• 0.5 = half orbit, 1 = full orbit, 2 = two orbits\n"
        "• Recommended: 1-5 orbits\n"
        "Your choice: "        
        )

        try:
            number_orbits = float(number_orbits)  # Try to convert to float ot int
        except ValueError:
            print('Wrong format, please type your value again')
    
            while True:
                try:
                    number_orbits = float(input("WRONG VALUE - Enter velocity as fraction of circular orbital speed: "))
                    break
                except ValueError:
                    print("Still not a valid number. Try again...")


        mass_earth  = input(
        "Enter Earth mass as fraction of actual mass:\n"
        "• 1.0 = actual Earth mass\n"
        "• <1.0 = reduced mass (e.g., 0.5 for half mass)\n"
        "• >1.0 = increased mass (e.g., 2.0 for double mass)\n"
        "• Recommended: 0.5-2.0\n"
        "Your choice: "
        )

        try:
            mass_earth = float(mass_earth)  # Try to convert to float ot int
        except ValueError:
            print('Wrong format, please type your value again')
    
            while True:
                try:
                    mass_earth = float(input("WRONG VALUE - Enter Earth mass as fraction of actual mass: "))
                    break
                except ValueError:
                    print("Still not a valid number. Try again...")

        velocity = input(
        "Enter the velocity as a fraction of the circular orbital speed.\n"
        "(e.g., 1 = exact circular orbit speed, values < 1 = elliptical orbit, "
        "values > 1 = the body may escape the orbit) "
        )

        try:
            velocity = float(velocity)  # Try to convert to float ot int
        except ValueError:
            print('Wrong format, please type your value again')
    
            while True:
                try:
                    velocity = float(input("WRONG VALUE - Enter the velocity as a fraction of the circular orbital speed: "))
                    break
                except ValueError:
                    print("Still not a valid number. Try again...")

        pm_distance = input(
       "Enter initial probe-Moon distance (km):\n"
        "• ~5000 = close approach (5,000 km)\n"
        "• ~10000 = medium distance (10,000 km)\n"
        "• ~90000 = far distance (90,000 km)\n"
        "• Recommended: 10000\n"
        "Your choice: "
        )

        
        try:
            pm_distance = float(pm_distance)  # Try to convert to float ot int
        except ValueError:
            print('Wrong format, please type your value again')
    
            while True:
                try:
                    pm_distance = float(input("WRONG VALUE - Enter the velocity as a fraction of the circular orbital speed: "))
                    break
                except ValueError:
                    print("Still not a valid number. Try again...")

        print("Running simulation... \n")


        simulation_earth_moon_probe(number_orbits, mass_earth, velocity, pm_distance)        
    elif MyInput != 'q':
        print('This is not a valid choice')
print('You have chosen to finish - goodbye.')

