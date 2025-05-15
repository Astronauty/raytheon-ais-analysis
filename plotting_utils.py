import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
import logging
from datetime import datetime, timedelta


def plot_single_ship_trajectory(t, mmsi):
    # Plot time vs x, y, theta, x_dot, y_dot, theta_dot
    plt.figure(figsize=(15, 10))

    # Plot x
    plt.subplot(2, 3, 1)
    plt.plot(single_ship_df['SecondsSinceStart'], single_ship_df['x'], label='x', color=colors[0])
    plt.xlabel('Time (s)')
    plt.ylabel('x (m)')
    plt.title('Time vs x')
    plt.grid()
    plt.legend()

    # Plot y
    plt.subplot(2, 3, 2)
    plt.plot(single_ship_df['SecondsSinceStart'], single_ship_df['y'], label='y', color=colors[1])
    plt.xlabel('Time (s)')
    plt.ylabel('y (m)')
    plt.title('Time vs y')
    plt.grid()
    plt.legend()

    # Plot theta
    plt.subplot(2, 3, 3)
    plt.plot(single_ship_df['SecondsSinceStart'], single_ship_df['theta'], label='theta', color=colors[2])
    plt.xlabel('Time (s)')
    plt.ylabel('theta (rad)')
    plt.title('Time vs theta')
    plt.grid()
    plt.legend()

    # Plot x_dot
    plt.subplot(2, 3, 4)
    plt.plot(single_ship_df['SecondsSinceStart'], single_ship_df['x_dot'], label='x_dot', color=colors[3])
    plt.xlabel('Time (s)')
    plt.ylabel('x_dot (m/s)')
    plt.title('Time vs x_dot')
    plt.grid()
    plt.legend()

    # Plot y_dot
    plt.subplot(2, 3, 5)
    plt.plot(single_ship_df['SecondsSinceStart'], single_ship_df['y_dot'], label='y_dot', color=colors[4])
    plt.xlabel('Time (s)')
    plt.ylabel('y_dot (m/s)')
    plt.title('Time vs y_dot')
    plt.grid()
    plt.legend()

    # Plot theta_dot
    plt.subplot(2, 3, 6)
    plt.plot(single_ship_df['SecondsSinceStart'], single_ship_df['theta_dot'], label='theta_dot', color=colors[5])
    plt.xlabel('Time (s)')
    plt.ylabel('theta_dot (rad/s)')
    plt.title('Time vs theta_dot')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()