import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import logging
from datetime import datetime, timedelta
import matplotlib.ticker as mticker


def plot_single_ship_state_trajectory(mmsi, t, state_trajectory):
    # Plot time vs x, y, theta, x_dot, y_dot, theta_dot

    # Use ScalarFormatter with scientific notation and 2 decimals
    # fmt = mticker.ScalarFormatter(useMathText=True)
    # fmt.set_powerlimits((-2, 2))
    # fmt.set_scientific(True)
    # fmt.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.2e}"))
    # fmt = mticker.FuncFormatter(lambda x, _: f"{x:.2e}")

    # Extract x, y, theta, x_dot, y_dot, theta_dot from state_trajectory
    x = state_trajectory[:, 0]
    y = state_trajectory[:, 1]
    theta = state_trajectory[:, 2]
    x_dot = state_trajectory[:, 3]
    y_dot = state_trajectory[:, 4]
    theta_dot = state_trajectory[:, 5]

    # Create a DataFrame for easier plotting
    single_ship_df = pd.DataFrame({
        'SecondsSinceStart': t,
        'x': x,
        'y': y,
        'theta': theta,
        'x_dot': x_dot,
        'y_dot': y_dot,
        'theta_dot': theta_dot
    })

    # Define colors for the plots
    colors = sns.color_palette("colorblind", 6)

    plt.figure(figsize=(15, 10))

    # Plot x
    plt.subplot(2, 3, 1)
    plt.plot(single_ship_df['SecondsSinceStart'], single_ship_df['x'], label='x', color=colors[0])
    plt.xlabel('Time (s)')
    plt.ylabel('$x$ (m)')
    # plt.title('Time vs x')
    plt.grid()
    plt.legend()
    # plt.gca().xaxis.set_major_formatter(fmt)
    # plt.gca().yaxis.set_major_formatter(fmt)


    # Plot y
    plt.subplot(2, 3, 2)
    plt.plot(single_ship_df['SecondsSinceStart'], single_ship_df['y'], label='y', color=colors[1])
    plt.xlabel('Time (s)')
    plt.ylabel('$y$ (m)')
    # plt.title('Time vs y')
    plt.grid()
    plt.legend()
    # plt.gca().xaxis.set_major_formatter(fmt)
    # plt.gca().yaxis.set_major_formatter(fmt)

    # Plot theta
    plt.subplot(2, 3, 3)
    plt.plot(single_ship_df['SecondsSinceStart'], single_ship_df['theta'], label='theta', color=colors[2])
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\dot{\theta}$ (rad)')
    # plt.title('Time vs theta')
    plt.grid()
    plt.legend()
    # plt.gca().xaxis.set_major_formatter(fmt)
    # plt.gca().yaxis.set_major_formatter(fmt)

    # Plot x_dot
    plt.subplot(2, 3, 4)
    plt.plot(single_ship_df['SecondsSinceStart'], single_ship_df['x_dot'], label='x_dot', color=colors[3])
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\dot{x}$ (m/s)')
    # plt.title('Time vs x_dot')
    plt.grid()
    plt.legend()
    # plt.gca().xaxis.set_major_formatter(fmt)
    # plt.gca().yaxis.set_major_formatter(fmt)

    # Plot y_dot
    plt.subplot(2, 3, 5)
    plt.plot(single_ship_df['SecondsSinceStart'], single_ship_df['y_dot'], label='y_dot', color=colors[4])
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\dot{y}$ (m/s)')
    plt.title('Time vs y_dot')
    plt.grid()
    plt.legend()
    # plt.gca().xaxis.set_major_formatter(fmt)
    # plt.gca().yaxis.set_major_formatter(fmt)

    # Plot theta_dot
    plt.subplot(2, 3, 6)
    plt.plot(single_ship_df['SecondsSinceStart'], single_ship_df['theta_dot'], label='theta_dot', color=colors[5])
    plt.xlabel('Time (s)')
    plt.ylabel(r'$\dot{\theta}$ (rad/s)')
    # plt.title('Time vs theta_dot')
    plt.grid()
    plt.legend()
    # plt.gca().xaxis.set_major_formatter(fmt)
    # plt.gca().yaxis.set_major_formatter(fmt)

    plt.tight_layout()
    plt.show()


def plot_single_ship_path(mmsi, t, state_trajectory):
    # Plot the path of the ship in the x-y plane
    plt.figure(figsize=(15, 10))
    plt.plot(state_trajectory[:, 0], state_trajectory[:, 1], marker='o', markersize=2)
    plt.title(f'Ship Path for MMSI {mmsi}')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.grid()
    plt.axis('equal')
    plt.show()