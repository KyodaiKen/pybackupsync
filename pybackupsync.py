#!/usr/bin/env python3
import os
import sys
import yaml
import time
import shutil
import signal
import argparse
import subprocess
import threading
import fnmatch
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty, PriorityQueue
import itertools

# --- Global State ---
SHUTDOWN_EVENT = threading.Event()
AUTO_MOUNTED_PATHS = [] # Tracks mount points we created
TERMINAL_WIDTH = 80
job_counter = itertools.count()

# --- Formatting Utilities ---

def get_terminal_width():
    try:
        return os.get_terminal_size().columns
    except OSError:
        return 80
    
def move_cursor_up(lines):
    """Moves the cursor up N lines and returns to the beginning of the line."""
    if lines > 0:
        # \033[NA moves the cursor up N lines. \r returns to the start of the line.
        return f"\r\033[{lines}A" 
    return ""

def format_bytes(size):
    # JEDEC 1.0: 1 KB = 1024 Bytes
    power = 1024
    n = size
    power_labels = {0 : '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
    count = 0
    while n >= power and count < 4:
        n /= power
        count += 1
    return f"{n:7.2f}{power_labels[count]}"

def format_rate(rate_bytes_per_sec):
    return f"{format_bytes(rate_bytes_per_sec)}/s"

def format_time(seconds):
    if seconds is None:
        return "--:--:--"
    if seconds > 86400 * 99: # > 99 days
        return "\033[91mETERNITY\033[0m" # Red Text
    
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    
    if d > 0:
        return f"{d:2d}d{h:2d}h{m:2d}m"
    return f" 0d{h:2d}h{m:2d}m"

def format_time_detailed(total_seconds_float):
    """
    Formats total seconds (float) into DDd HHh MMm SSs MSms format.
    """
    if total_seconds_float is None or total_seconds_float < 0:
        return "N/A"
        
    total_milliseconds = int(round(total_seconds_float * 1000))

    ms = total_milliseconds % 1000
    total_seconds = total_milliseconds // 1000

    s = total_seconds % 60
    total_minutes = total_seconds // 60

    m = total_minutes % 60
    total_hours = total_minutes // 60

    h = total_hours % 24
    d = total_hours // 24

    return (
        f"{d:02d}d "
        f"{h:02d}h "
        f"{m:02d}m "
        f"{s:02d}s "
        f"{ms:03d}ms"
    )

def shorten_path(path_str, max_len):
    if len(path_str) <= max_len:
        return path_str
    half = (max_len - 3) // 2
    return path_str[:half] + "..." + path_str[-half:]

def draw_progress_bar(percent, width):
    if width < 5: return "[]"
    bar_width = width - 2
    filled = int(bar_width * (percent / 100.0))
    bar_str = "=" * filled
    if filled < bar_width:
        bar_str += ">"
    padding = " " * (bar_width - len(bar_str))
    return f"[{bar_str}{padding}]"

# --- Drive & Path Management ---

def safe_join(base, *paths):
    """Safely join paths ensuring we don't escape the base."""
    # We strip leading slashes from relative parts to ensure join works as expected
    clean_paths = [p.lstrip("/") for p in paths]
    return os.path.join(base, *clean_paths)

def get_mount_point_from_uuid(uuid):
    """Returns mount point if UUID is currently mounted, else None."""
    # Resolve the UUID symlink to a real device (e.g., /dev/sda1)
    dev_path = f"/dev/disk/by-uuid/{uuid}"
    if not os.path.exists(dev_path):
        return None
    
    real_dev = os.path.realpath(dev_path)
    
    with open('/proc/mounts', 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                m_dev = parts[0]
                m_point = parts[1]
                # Check if device matches (resolving symlinks for the mount entry too)
                if m_dev == real_dev or (os.path.exists(m_dev) and os.path.realpath(m_dev) == real_dev):
                    return m_point
    return None

def mount_drive(uuid, mount_point):
    """Mounts drive if not mounted. Returns success path or None."""
    dev_path = f"/dev/disk/by-uuid/{uuid}"
    
    # 1. Check if physically present
    if not os.path.exists(dev_path):
        print(f"\n>> Please connect drive {mount_point}...")
        while not os.path.exists(dev_path):
            if SHUTDOWN_EVENT.is_set(): return None
            time.sleep(1)
            
    # 2. Check if already mounted
    existing_mp = get_mount_point_from_uuid(uuid)
    if existing_mp:
        return existing_mp

    # 3. Mount
    print(f"Mounting {uuid} at {mount_point}...")
    os.makedirs(mount_point, exist_ok=True)
    try:
        subprocess.run(["mount", f"UUID={uuid}", mount_point], check=True)
        AUTO_MOUNTED_PATHS.append(mount_point)
        return mount_point
    except subprocess.CalledProcessError as e:
        print(f"Failed to mount {uuid}: {e}")
        return None

def unmount_all(prompt=False):
    if not AUTO_MOUNTED_PATHS:
        return

    if prompt:
        q = input(f"\nUnmount automatically mounted drives? ({len(AUTO_MOUNTED_PATHS)}) [Y/n]: ")
        if q.lower() == 'n':
            return

    print("\nSynchronizing buffers (sync)...")
    try:
        subprocess.run(["sync"], check=True)
    except:
        pass

    for mp in reversed(AUTO_MOUNTED_PATHS):
        print(f"Unmounting {mp}...")
        subprocess.run(["umount", "-l", mp])
        # Wait until it's actually gone from /proc/mounts
        while os.path.ismount(mp):
            time.sleep(0.5)
            
    AUTO_MOUNTED_PATHS.clear()

# --- File Operations & Space Management ---

def scan_source(root_path, dest_root, exclude_patterns):
    """
    Scans source recursively. 
    root_path: The absolute path (Mount + Relative Config).
    Returns list of (absolute_path, relative_to_root, size) for files that don't exist in dest_root.
    """
    files_found = []
    total_bytes_needed = 0
    
    if not os.path.exists(root_path):
        print(f"Error: Source path does not exist: {root_path}")
        return [], 0

    print(f"Scanning source: {root_path}")
    
    for root, dirs, filenames in os.walk(root_path):
        # Filter directories
        dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, p) for p in exclude_patterns)]
        
        for f in filenames:
            if any(fnmatch.fnmatch(f, p) for p in exclude_patterns):
                continue

            abs_path = os.path.join(root, f)
            try:
                new_size = os.path.getsize(abs_path)
                rel_path = os.path.relpath(abs_path, root_path)
                dest_file = os.path.join(dest_root, rel_path)
                existing_size = 0 # Assume 0 if file doesn't exist
                if os.path.exists(dest_file):
                    existing_size = os.path.getsize(dest_file)
                    if new_size == existing_size:
                        continue
                # Calculate the net difference
                net_change = new_size - existing_size
                files_found.append((abs_path, rel_path, new_size, existing_size))
                total_bytes_needed += net_change
            except OSError:
                pass
                
    return files_found, total_bytes_needed

def get_files_with_ctime(directory):
    """Return list of (path, ctime) for all files in directory recursively."""
    res = []
    for root, _, files in os.walk(directory):
        for f in files:
            p = os.path.join(root, f)
            try:
                res.append((p, os.path.getctime(p)))
            except OSError:
                pass
    return res

def ensure_space(target_root, needed_bytes, policy):
    """
    Deletes files in target_root based on policy until needed_bytes fits.
    target_root: The specific destination folder (Mount + Dest Relative).
    """
    if not os.path.exists(target_root):
        # If the folder doesn't exist, we assume the partition has space, 
        # or we check the partition of the parent.
        os.makedirs(target_root, exist_ok=True)

    # Check partition usage
    total, used, free = shutil.disk_usage(target_root)
    
    if needed_bytes <= 0 or free >= needed_bytes:
        return

    deficit = needed_bytes - free
    print(f"Cleaning up {target_root} (Policy: {policy}). Need {format_bytes(deficit)} more.")

    files = get_files_with_ctime(target_root)
    to_delete = []

    if policy == "oldest":
        # Sort by ctime asc
        files.sort(key=lambda x: x[1])
        to_delete = [x[0] for x in files]

    elif policy == "month_dupes":
        # Group by YYYY-MM
        groups = {}
        for p, t in files:
            m = datetime.fromtimestamp(t).strftime('%Y-%m')
            if m not in groups: groups[m] = []
            groups[m].append((p, t))
        
        candidates = []
        for m in groups:
            # Sort files within month
            month_files = sorted(groups[m], key=lambda x: x[1])
            # Keep the last one (newest), mark others for deletion
            if len(month_files) > 1:
                candidates.extend(month_files[:-1])
        
        # Sort candidates by age to delete oldest duplicates first
        candidates.sort(key=lambda x: x[1])
        to_delete = [x[0] for x in candidates]

    # Execute deletion
    deleted_size = 0
    for fpath in to_delete:
        try:
            sz = os.path.getsize(fpath)
            os.remove(fpath)
            print(f"Delted {fpath}")
            deleted_size += sz
            free += sz
            if free >= needed_bytes:
                print(f"Freed enough space ({format_bytes(deleted_size)}).")
                return
        except OSError:
            pass

    print(f"Warning: Cleanup finished but might still lack space.")

# --- Copy Logic & Workers ---

class Job:
    def __init__(self, src, dst, size, priority, dest_uuid):
        self.src = src
        self.dst = dst
        self.size = size
        self.priority = priority
        self.dest_uuid = dest_uuid  # Store the destination UUID
        self.count = next(job_counter) # Unique counter for stable sort
        
        # Stats
        self.bytes_done = 0
        self.rate = 0.0
        self.avg_rate = 0.0
        self.eta = None
        self.completed = False
        self.error = None
        
        # Resume detection
        self.resume_mode = False
        """ NOT IMPLEMENTED
            if os.path.exists(dst):
            dst_sz = os.path.getsize(dst)
            if dst_sz < size:
                self.resume_mode = True
                self.bytes_done = dst_sz """

    def __lt__(self, other):
        # 1. Primary sort: Lower priority value comes first (higher priority)
        if self.priority != other.priority:
            return self.priority < other.priority
        
        # 2. Secondary sort (Interleaving): Use the UUID string for alternating
        #    This ensures Dest A jobs come before Dest B jobs, creating a stable,
        #    alternating order when priorities are equal.
        if self.dest_uuid != other.dest_uuid:
            return self.dest_uuid < other.dest_uuid
            
        # 3. Tertiary sort: If priority and UUID are equal, use insertion order.
        return self.count < other.count

def worker_func(job, queue):
    """
    Executes PV.
    Command: pv -i 0.5 -F "%t %b %r %a" -n {$source} > {$destination}
    "-L 1048576" is for testing the progress bar to slow it down to 1048576 byte per second. (WOW)
    Add -s if resuming.
    """
    cmd = ["pv", "-i", "0", "-F", "%t %b %r %a", "-n", job.src, "-o", job.dst]
    
    #if job.resume_mode:
        # NOT IMPLEMENTED YET

    # For testing
    # cmd.insert(1, "-L 1048576")

    try:
        # PV with -n writes numeric data to Stderr
        proc = subprocess.Popen(
            cmd, 
            stderr=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            universal_newlines=True,
            bufsize=1
        )
        
        for line in proc.stderr:
            if SHUTDOWN_EVENT.is_set():
                proc.terminate()
                break
            
            parts = line.strip().split()
            # Expecting 4 parts based on format "%t %b %r %a"
            # %t: elapsed, %b: bytes, %r: rate, %a: avg rate
            # Note: format string logic in PV usually produces space separated values.
            if len(parts) >= 4:
                try:
                    # Update Job
                    job.bytes_done = int(parts[1])
                    job.rate = float(parts[2])
                    job.avg_rate = float(parts[3])
                    
                    if job.avg_rate > 0:
                        rem = job.size - job.bytes_done
                        job.eta = int(rem / job.avg_rate) if rem > 0 else 0
                    
                    queue.put(('progress', job))
                except ValueError:
                    pass
        
        proc.wait()
        
        if proc.returncode == 0:
            job.completed = True
            job.bytes_done = job.size # Ensure 100% visual
            queue.put(('done', job))
        else:
            job.error = f"Exit Code {proc.returncode}"
            #print(f"Error code: {proc.returncode}, command: {cmd}");
            queue.put(('error', job))

    except Exception as e:
        job.error = str(e)
        queue.put(('error', job))

def display_loop(jobs, max_parallel, total_scope_bytes):
    q = Queue()
    
    # Priority Queue for pending
    pending = PriorityQueue()
    for j in jobs:
        pending.put(j)
        
    active = []
    active_dest_uuids = set()
    
    def refill_workers():
        # Holding area for jobs that couldn't launch because their drive was busy
        deferred_jobs = [] 
        
        while len(active) < max_parallel and not pending.empty():
            job = pending.get()
            
            if job.dest_uuid not in active_dest_uuids:
                # LAUNCH JOB: Destination is free
                active.append(job)
                active_dest_uuids.add(job.dest_uuid)
                
                t = threading.Thread(target=worker_func, args=(job, q))
                t.daemon = True
                t.start()
            else:
                # DEFER JOB: Destination is busy, put it aside temporarily
                deferred_jobs.append(job)
        
        # Put any deferred jobs back into the main queue
        for job in deferred_jobs:
            pending.put(job) # Put deferred jobs back in the PriorityQueue

    refill_workers()
    
    # Tracking totals
    completed_jobs = []
    lines_printed_last_cycle = 0

    try:
        while not SHUTDOWN_EVENT.is_set():
            # Process queue updates non-blocking
            try:
                while True:
                    kind, job_obj = q.get_nowait()
                    if kind in ('done', 'error'):
                        if job_obj in active:
                            active.remove(job_obj)
                            completed_jobs.append(job_obj)
                            
                            if job_obj.dest_uuid in active_dest_uuids:
                                active_dest_uuids.remove(job_obj.dest_uuid)

                            refill_workers() # Attempt to launch the next available job
            except Empty:
                pass

            if not active and pending.empty():
                break

            # Rendering
            width = get_terminal_width()
            #print("\033[H\033[J", end="") # ANSI Clear Screen

            sys.stdout.write(move_cursor_up(lines_printed_last_cycle))
            sys.stdout.flush()

            output_lines = []
            output_lines.append("+" + ("=" * (width - 2)) + "+")

            # --- Individual Jobs ---
            for j in active:
                s_name = shorten_path(j.src, width - 12)
                d_name = shorten_path(j.dst, width - 12)
                
                pct = 0.0
                if j.size > 0: pct = (j.bytes_done / j.size) * 100.0
                
                # Format: {$PercentDone} {$BytesTransferred} / {$file_size} @{TransferRate} ({$TransferRateAvg}) ETA: {$TimeRemaining}
                stats = f"{format_bytes(j.bytes_done)}/{format_bytes(j.size)}={pct:6.2f}% @{format_rate(j.rate)} a{format_rate(j.avg_rate)} ETA: {format_time(j.eta)} "
                
                bar_space = width - len(stats) - 12
                bar = draw_progress_bar(pct, bar_space)
                
                # Calculate padding needed for the Source/Destination lines
                s_pad = " " * (width - len(s_name) - 12)
                d_pad = " " * (width - len(d_name) - 12)

                output_lines.append(f"| Source: {s_name}{s_pad} |")
                output_lines.append(f"| Destin: {d_name}{d_pad} |")
                output_lines.append(f"| Prgrss: {stats}{bar} |")
                output_lines.append("+" + ("-" * (width - 2)) + "+")

            # --- Total Progress ---
            # Sum bytes done from (completed + active)
            # Pending bytes are 0 done.
            
            # Recalculate accurate total bytes done
            total_done = sum(j.size for j in completed_jobs) + sum(j.bytes_done for j in active)
            
            # Total Rate: Sum of active current rates
            total_rate = sum(j.rate for j in active)
            # Total Avg Rate: Sum of active average rates (approximation requested)
            total_avg_rate = sum(j.avg_rate for j in active)
            
            total_pct = 0.0
            if total_scope_bytes > 0:
                total_pct = (total_done / total_scope_bytes) * 100.0
                
            # Total ETA
            total_eta = 0
            remaining_total = total_scope_bytes - total_done
            if total_avg_rate > 0:
                total_eta = remaining_total / total_avg_rate

            t_stats = f"{format_bytes(total_done)}/{format_bytes(total_scope_bytes)}={total_pct:6.2f}% @{format_rate(total_rate)} a{format_rate(total_avg_rate)} ETA: {format_time(total_eta)} "
            t_bar = draw_progress_bar(total_pct, width - len(t_stats) - 12)
            
            output_lines.append(f"| TOTAL : {t_stats}{t_bar} |")
            output_lines.append("+" + ("=" * (width - 2)) + "+")

            # Print the new content
            sys.stdout.write("\n".join(output_lines))
            
            # Clear extra lines if the number of active jobs decreased
            current_line_count = len(output_lines)
            if lines_printed_last_cycle > current_line_count:
                extra_lines = lines_printed_last_cycle - current_line_count
               
                # Move cursor up to the start of the extra space
                sys.stdout.write(move_cursor_up(extra_lines))
               
                # Overwrite the old content with blank lines
                sys.stdout.write(("\r" + " " * width + "\n") * extra_lines)
               
                # Move cursor back up to the start of the footer
                sys.stdout.write(move_cursor_up(extra_lines))

            lines_printed_last_cycle = current_line_count - 1
            
            sys.stdout.flush()
            
            time.sleep(0.5)

    except KeyboardInterrupt:
        SHUTDOWN_EVENT.set()

    finally:
        # Clear the display area cleanly when finished or interrupted
        if lines_printed_last_cycle > 0:
            sys.stdout.write(move_cursor_up(lines_printed_last_cycle))
            sys.stdout.write(" " * width + "\n" * (lines_printed_last_cycle + 1))
            sys.stdout.write(move_cursor_up(lines_printed_last_cycle))
            sys.stdout.flush()

# --- Main Logic ---

def main():
    start_time = time.time()

    signal.signal(signal.SIGINT, lambda s, f: SHUTDOWN_EVENT.set())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-um", "--umount", action="store_true", help="Unmount after success")
    parser.add_argument("-C", "--config", help="Config file path")
    args = parser.parse_args()

    # Load Config
    cfg_path = args.config if args.config else os.path.expanduser("~/.pybackupsync.yaml")
    if not os.path.exists(cfg_path):
        print("Config file not found.")
        return
    
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    # Mount Source & Define Source Root
    src_cfg = config['source']
    src_mp = mount_drive(src_cfg['uuid'], src_cfg['mount_point'])
    if not src_mp: return

    # Combine Mount Point + Relative Path
    src_root = safe_join(src_mp, src_cfg.get('path', ''))

    # Mount Destinations & Check Space
    dest_cfgs = config['destinations']
    jobs_by_file = {} 
    total_job_bytes = 0
    
    # Calculate global total bytes to transfer (Source Size * Num Valid Destinations)
    # We build the job list to know exactly what runs.
    
    for d_cfg in dest_cfgs:
        d_mp = mount_drive(d_cfg['uuid'], d_cfg['mount_point'])
        if not d_mp:
            print(f"Skipping destination {d_cfg['mount_point']}")
            continue
            
        # Define specific destination root
        dest_root = safe_join(d_mp, d_cfg.get('path', ''))

        # Determine files to be copied
        files_to_copy_data, total_net_change_bytes = scan_source(src_root, dest_root, d_cfg.get('deletion_exclude_patterns', []))

        if not files_to_copy_data:
            print(f"Destination {d_cfg['mount_point']} is up to date.")
            continue
        
        # Check space on this specific root's partition
        ensure_space(dest_root, total_net_change_bytes, d_cfg.get('deletion_policy', 'oldest'))
    
        # Generate Jobs (Note: we use new_size as the job size)
        for src_abs, rel_path, new_size, existing_size in files_to_copy_data:
            dst_abs = os.path.join(dest_root, rel_path)
            
            # Create directories
            os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
            
            # We pass new_size to the Job
            jobs_by_file.setdefault(rel_path, []).append(Job(src_abs, dst_abs, new_size, d_cfg.get('priority', 99), dest_uuid=d_cfg['uuid']))
            total_job_bytes += new_size # Keep accumulating the final *destination* size for global stats

    if not jobs_by_file:
        print("No jobs created.")
        unmount_all(not args.umount)
        return
    
    final_jobs_list = []
    sorted_file_keys = sorted(jobs_by_file.keys())

    # Iterate over files (File 1, File 2, File 3...)
    for rel_path in sorted_file_keys:
        # Iterate over the destinations for that file (Dest A, Dest B, Dest C...)
        # This guarantees the first N jobs started are all for the first source file 
        # but target different destinations.
        
        # We sort the jobs here by priority to respect that first for a given file
        jobs_for_file = sorted(jobs_by_file[rel_path], key=lambda j: j.priority)
        
        final_jobs_list.extend(jobs_for_file)

    # Run Copy
    print(f"Starting Copy. Total data: {format_bytes(total_job_bytes)}")
    time.sleep(1)
    
    display_loop(final_jobs_list, config.get('max_parallel_processes', 2), total_job_bytes)
    
    end_time = time.time()
    end_time_str = format_time_detailed(end_time - start_time)

    print(f"\nCopy operation finished after {end_time_str}.")
    
    # Cleanup
    if args.umount:
        unmount_all()
    elif AUTO_MOUNTED_PATHS:
        unmount_all(prompt=True)

if __name__ == "__main__":
    main()