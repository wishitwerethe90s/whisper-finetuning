#!/usr/bin/env python
# memory_monitor.py - Run in parallel with the main script to monitor memory usage

import psutil
import time
import os
import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np

def monitor_memory(pid=None, interval=5, output_dir="memory_logs"):
    """
    Monitor memory usage of a process and log it
    
    Args:
        pid: Process ID to monitor (None = current process)
        interval: Interval in seconds between measurements
        output_dir: Directory to save logs and plots
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get process
    if pid is None:
        pid = os.getpid()
    
    # Try to get process info
    try:
        process = psutil.Process(pid)
        process_name = process.name()
        print(f"Monitoring process {pid} ({process_name})")
    except psutil.NoSuchProcess:
        print(f"Process {pid} not found. Please check the PID.")
        return
    
    # Prepare log file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"memory_log_{pid}_{timestamp}.csv")
    plot_file = os.path.join(output_dir, f"memory_plot_{pid}_{timestamp}.png")
    
    # Write CSV header
    with open(log_file, "w") as f:
        f.write("timestamp,elapsed_seconds,rss_mb,vms_mb,cpu_percent,system_memory_percent\n")
    
    # Initialize data arrays for plotting
    times = []
    rss_values = []
    vms_values = []
    cpu_values = []
    sys_mem_values = []
    
    start_time = time.time()
    print(f"Starting memory monitoring. Logs will be saved to {log_file}")
    print("Press Ctrl+C to stop monitoring.")
    
    try:
        while True:
            # Get current time
            current_time = time.time()
            elapsed = current_time - start_time
            timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            try:
                # Get memory info
                mem_info = process.memory_info()
                rss_mb = mem_info.rss / (1024 * 1024)
                vms_mb = mem_info.vms / (1024 * 1024)
                
                # Get CPU usage
                cpu_percent = process.cpu_percent(interval=0.1)
                
                # Get system memory usage
                system_memory = psutil.virtual_memory()
                system_memory_percent = system_memory.percent
                
                # Log data
                with open(log_file, "a") as f:
                    f.write(f"{timestamp_str},{elapsed:.2f},{rss_mb:.2f},{vms_mb:.2f},{cpu_percent:.2f},{system_memory_percent:.2f}\n")
                
                # Store data for plotting
                times.append(elapsed / 60)  # Convert to minutes
                rss_values.append(rss_mb)
                vms_values.append(vms_mb)
                cpu_values.append(cpu_percent)
                sys_mem_values.append(system_memory_percent)
                
                # Print status
                print(f"[{timestamp_str}] Elapsed: {elapsed:.2f}s, RSS: {rss_mb:.2f}MB, VMS: {vms_mb:.2f}MB, CPU: {cpu_percent:.2f}%, System Mem: {system_memory_percent:.2f}%")
                
                # Update plot periodically (every 10 measurements)
                if len(times) % 10 == 0:
                    create_plot(times, rss_values, vms_values, cpu_values, sys_mem_values, plot_file)
                
            except psutil.NoSuchProcess:
                print(f"Process {pid} no longer exists. Stopping monitoring.")
                break
            
            # Sleep until next interval
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
    
    # Create final plot
    create_plot(times, rss_values, vms_values, cpu_values, sys_mem_values, plot_file)
    print(f"Memory monitoring completed. Logs saved to {log_file} and plot to {plot_file}")

def create_plot(times, rss_values, vms_values, cpu_values, sys_mem_values, plot_file):
    """Create a memory usage plot"""
    if not times:
        return
        
    plt.figure(figsize=(12, 8))
    
    # Memory subplot
    plt.subplot(2, 1, 1)
    plt.plot(times, rss_values, 'b-', label='RSS Memory')
    plt.plot(times, vms_values, 'r--', label='Virtual Memory')
    plt.title('Process Memory Usage Over Time')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Memory (MB)')
    plt.grid(True)
    plt.legend()
    
    # CPU and System Memory subplot
    plt.subplot(2, 1, 2)
    plt.plot(times, cpu_values, 'g-', label='CPU Usage')
    plt.plot(times, sys_mem_values, 'm--', label='System Memory Usage')
    plt.title('CPU and System Memory Usage')
    plt.xlabel('Time (minutes)')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor memory usage of a process")
    parser.add_argument("--pid", type=int, help="Process ID to monitor")
    parser.add_argument("--interval", type=int, default=5, help="Interval between measurements in seconds")
    parser.add_argument("--output-dir", default="memory_logs", help="Directory to save logs and plots")
    
    args = parser.parse_args()
    
    monitor_memory(args.pid, args.interval, args.output_dir)