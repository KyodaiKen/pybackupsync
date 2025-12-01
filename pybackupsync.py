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
SLOW = False
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
        return "(unknown)"
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
    half = (max_len - 1) // 2
    return path_str[:half] + "…" + path_str[-half:]

def draw_progress_bar(percent, width):
    if width < 5: return ""
    bar_width = width
    filled = int(bar_width * (percent / 100.0))
    bar_str = "█" * filled
    #if filled < bar_width:
    #    bar_str += "▒"
    padding = "░" * (bar_width - len(bar_str))
    return f"{bar_str}{padding}"

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

def optimize_drive_if_ssd(mount_point):
    """
    Checks if the drive mounted at mount_point is an SSD (non-rotational).
    If so, runs fstrim to optimize free space.
    """
    try:
        # 1. Find the source device for the mount point
        # findmnt -n -o SOURCE --target /path/to/mount
        result = subprocess.run(
            ["findmnt", "-n", "-o", "SOURCE", "--target", mount_point],
            capture_output=True, text=True, check=True
        )
        device_node = result.stdout.strip()
        
        # 2. Check rotational status
        # lsblk -dno ROTATIONAL /dev/sdX
        # Returns 1 for HDD (rotational), 0 for SSD
        res_rot = subprocess.run(
            ["lsblk", "-dno", "ROTATIONAL", device_node],
            capture_output=True, text=True, check=True
        )
        is_ssd = (res_rot.stdout.strip() == "0")
        
        if is_ssd:
            print(f"SSD detected at {mount_point}. Running fstrim...")
            subprocess.run(["fstrim", "-v", mount_point], check=True)
        else:
            # print(f"HDD detected at {mount_point}. Skipping fstrim.")
            pass
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not check/optimize SSD for {mount_point}: {e}")

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

def get_files_in_specific_directories(directory_list):
    """
    Return list of (path, ctime) for all files ONLY in the provided directories.
    This does NOT scan recursively. It only looks at the specific folders
    where new files are being copied to.
    """
    res = []
    
    # Use a set to avoid scanning the same directory twice if multiple files go there
    unique_dirs = set(directory_list)
    
    for d in unique_dirs:
        if not os.path.exists(d):
            continue
            
        try:
            # We use scandir for better performance on flat directory scans
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        try:
                            # Get status directly from entry
                            stat = entry.stat()
                            res.append((entry.path, stat.st_ctime))
                        except OSError:
                            pass
        except OSError:
            pass
            
    return res

def ensure_space(mount_point, target_dirs, needed_bytes, policy):
    """
    Deletes files in target_dirs based on policy until needed_bytes fits.
    mount_point: Used to check the actual disk partition usage.
    target_dirs: A list of specific directories where we are allowed to delete old files.
    """
    # Check partition usage on the mount point
    total, used, free = shutil.disk_usage(mount_point)
    
    if needed_bytes <= 0 or free >= needed_bytes:
        return

    deficit = needed_bytes - free
    print(f"Cleaning up (Policy: {policy}). Need {format_bytes(deficit)} more.")

    # Get all candidate files in the specified directories
    files = get_files_in_specific_directories(target_dirs)
    
    # List to hold paths selected for deletion
    to_delete = []

    if policy == "oldest":
        # Sort by ctime asc (oldest first)
        files.sort(key=lambda x: x[1])
        to_delete = [x[0] for x in files]
        
    elif policy == "month_dupes":
        
        # --- Stage 1: Identify and prioritize monthly duplicates for deletion ---
        groups = {}
        for p, t in files:
            # Group by YYYY-MM
            m = datetime.fromtimestamp(t).strftime('%Y-%m')
            if m not in groups: groups[m] = []
            groups[m].append((p, t))
        
        duplicates_to_delete = []
        remaining_files = [] # Files that survived the duplicate removal (max 1 per month)
        
        for m in groups:
            # Sort files within month (oldest to newest)
            month_files = sorted(groups[m], key=lambda x: x[1])
            
            # Keep the last one (newest), mark others for deletion
            if len(month_files) > 1:
                duplicates_to_delete.extend(month_files[:-1])
                remaining_files.append(month_files[-1]) # The keeper
            else:
                remaining_files.append(month_files[0]) # The single archive
        
        # Prioritize deletion of monthly duplicates (oldest first)
        duplicates_to_delete.sort(key=lambda x: x[1])
        
        # We will now delete these files sequentially until space is freed or we run out.
        to_delete = [x[0] for x in duplicates_to_delete]

    # --- EXECUTE DELETION LOOP ---
    deleted_size = 0
    
    # 1. Execute deletions based on the initial 'to_delete' list (used by both policies)
    paths_to_remove_from_candidates = [] # Tracks files deleted in Stage 1 for 'month_dupes'

    for i, fpath in enumerate(to_delete):
        if free >= needed_bytes:
            break
        try:
            sz = os.path.getsize(fpath)
            os.remove(fpath)
            print(f"Deleted (Policy: {policy.upper()}): {fpath}")
            deleted_size += sz
            free += sz
            if policy == "month_dupes":
                paths_to_remove_from_candidates.append(fpath)
        except OSError:
            pass
            
    # 2. MONTH_DUPES ONLY: If space is still needed, delete oldest monthly archives
    if policy == "month_dupes" and free < needed_bytes:
        print(f"Monthly duplicates cleaned. Still need {format_bytes(needed_bytes - free)} more. Starting monthly archive deletion.")
        
        # Filter 'remaining_files' to ensure we only have files that WEREN'T duplicates
        # and then delete the oldest of those until space is freed.
        
        # Sort the remaining single monthly files by age (oldest first)
        remaining_files.sort(key=lambda x: x[1]) 
        
        # Continue deletion using the oldest remaining archive files
        for path, _ in remaining_files:
            # Ensure this file wasn't already in the duplicates list (edge case safety)
            if path in paths_to_remove_from_candidates:
                continue
                
            if free >= needed_bytes:
                break
            try:
                sz = os.path.getsize(path)
                os.remove(path)
                print(f"Deleted (Archive): {path}")
                deleted_size += sz
                free += sz
            except OSError:
                pass

    if deleted_size > 0:
        # Optimize SSD after deletion
        optimize_drive_if_ssd(mount_point)

    if free >= needed_bytes:
        print(f"Freed enough space ({format_bytes(deleted_size)}).")
    else:
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

    def __lt__(self, other):
        # 1. Primary sort: Lower priority value comes first (higher priority)
        if self.priority != other.priority:
            return self.priority < other.priority
        
        # 2. Secondary sort (Interleaving): Use the UUID string for alternating
        if self.dest_uuid != other.dest_uuid:
            return self.dest_uuid < other.dest_uuid
            
        # 3. Tertiary sort: If priority and UUID are equal, use insertion order.
        return self.count < other.count

def worker_func(job, queue):
    global SLOW
    """
    Executes PV.
    Command: pv -i 0 -F "%t %b %r %a" -n {$source} -o {$destination}
    """
    cmd = ["pv", "-i", "0", "-F", "%t %b %r %a", "-n", job.src, "-o", job.dst]
    if SLOW:
        cmd.insert(3, f"-L {(1024*1024)}")
    
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

            sys.stdout.write(move_cursor_up(lines_printed_last_cycle))
            sys.stdout.flush()

            output_lines = []
            output_lines.append("┏" + ("━" * (width - 2)) + "┓")

            # --- Individual Jobs ---
            first = True
            for j in active:
                s_name = shorten_path(j.src, width - 12)
                d_name = shorten_path(j.dst, width - 12)
                
                pct = 0.0
                if j.size > 0: pct = (j.bytes_done / j.size) * 100.0
                
                stats = f"{format_bytes(j.bytes_done)}/{format_bytes(j.size)}={pct:6.2f}% @{format_rate(j.rate)} a{format_rate(j.avg_rate)} ETA: {format_time(j.eta)} "
                
                bar_space = width - len(stats) - 12
                bar = draw_progress_bar(pct, bar_space)
                
                # Calculate padding needed for the Source/Destination lines
                s_pad = " " * (width - len(s_name) - 12)
                d_pad = " " * (width - len(d_name) - 12)

                if first == False:
                    output_lines.append("┠" + ("─" * (width - 2)) + "┨")
                output_lines.append(f"┃ Source: {s_name}{s_pad} ┃")
                output_lines.append(f"┃ Destin: {d_name}{d_pad} ┃")
                output_lines.append(f"┃ Prgrss: {stats}{bar} ┃")
                
                first = False

            # --- Total Progress ---
            # Recalculate accurate total bytes done
            total_done = sum(j.size for j in completed_jobs) + sum(j.bytes_done for j in active)
            
            total_rate = sum(j.rate for j in active)
            total_avg_rate = sum(j.avg_rate for j in active)
            
            total_pct = 0.0
            if total_scope_bytes > 0:
                total_pct = (total_done / total_scope_bytes) * 100.0
                
            total_eta = 0
            remaining_total = total_scope_bytes - total_done
            if total_avg_rate > 0:
                total_eta = remaining_total / total_avg_rate

            t_stats = f"{format_bytes(total_done)}/{format_bytes(total_scope_bytes)}={total_pct:6.2f}% @{format_rate(total_rate)} a{format_rate(total_avg_rate)} ETA: {format_time(total_eta)} "
            t_bar = draw_progress_bar(total_pct, width - len(t_stats) - 12)
            
            output_lines.append("┣" + ("━" * (width - 2)) + "┫")
            output_lines.append(f"┃ TOTAL : {t_stats}{t_bar} ┃")
            output_lines.append("┗" + ("━" * (width - 2)) + "┛")

            # Print the new content
            sys.stdout.write("\n".join(output_lines))
            
            # Clear extra lines if the number of active jobs decreased
            current_line_count = len(output_lines) - 1
            lines_to_clear = lines_printed_last_cycle - current_line_count
            
            if lines_to_clear > 0:
                lines_to_clear += 1
                sys.stdout.write(("\r" + (" " * width) + "\n") * lines_to_clear)
                sys.stdout.write(move_cursor_up(lines_to_clear))

            # Update the count for the next iteration. This must be the actual height of the current frame minus 1 (because the top line of the frame is the start of the frame).
            lines_printed_last_cycle = current_line_count
            sys.stdout.flush()
            time.sleep(0.1)

    except KeyboardInterrupt:
        SHUTDOWN_EVENT.set()

    finally:
        # Move past the final progress bar
        sys.stdout.write("\n")
        sys.stdout.flush()
        
        # Clear the whole block one last time to leave a clean terminal
        if lines_printed_last_cycle > 0:
            # Calculate total height of the last frame drawn
            total_height = lines_printed_last_cycle + 1
            
            # 1. Move up to the start of the block
            sys.stdout.write(move_cursor_up(total_height))
            
            # 2. Overwrite everything with whitespace
            sys.stdout.write(("\r" + (" " * width) + "\n") * total_height)
            
            # 3. Move back up to leave the cursor where the block started
            #    (so the prompt returns at the correct place)
            sys.stdout.write(move_cursor_up(total_height))
            sys.stdout.flush()

# --- Main Logic ---

def main():
    global SLOW
    start_time = time.time()
    signal.signal(signal.SIGINT, lambda s, f: SHUTDOWN_EVENT.set())
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-um", "--umount", action="store_true", help="Unmount after success")
    parser.add_argument("-C", "--config", help="Config file path")
    parser.add_argument("-s", "--slow", action="store_true", help="Limit the copy speed to 0.5 MB/s for testing")
    args = parser.parse_args()

    SLOW = args.slow

    if SLOW:
        print("WARNING: SLOW MODE ENABLED -> 1MByte/s speed limit enabled!")

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
    
    for d_cfg in dest_cfgs:
        d_mp = mount_drive(d_cfg['uuid'], d_cfg['mount_point'])
        if not d_mp:
            print(f"Skipping destination {d_cfg['mount_point']}")
            continue
            
        dest_root = safe_join(d_mp, d_cfg.get('path', ''))
        deletion_patterns = d_cfg.get('deletion_exclude_patterns', [])
        deletion_disabled = (deletion_patterns == ['*'])

        # Determine files to be copied
        files_to_copy_data, total_net_change_bytes = scan_source(src_root, dest_root, src_cfg.get('exclude_patterns', []))

        if not files_to_copy_data:
            print(f"Destination {d_cfg['mount_point']} is up to date.")
            continue
        
        # Gather list of specific destination directories involved
        affected_directories = set()
        for _, rel_path, _, _ in files_to_copy_data:
            # We want the folder containing the file in the destination
            full_dest_path = os.path.join(dest_root, rel_path)
            affected_directories.add(os.path.dirname(full_dest_path))

        # Check space
        if deletion_disabled:
            print(f"Cleanup skipped for {d_cfg['mount_point']} (Deletion disabled by '*' pattern).")
        else:
            # Pass mount_point (for total partition check) AND specific directories (for deletion candidates)
            ensure_space(d_mp, list(affected_directories), total_net_change_bytes, d_cfg.get('deletion_policy', 'oldest'))

        # Generate Jobs
        for src_abs, rel_path, new_size, existing_size in files_to_copy_data:
            dst_abs = os.path.join(dest_root, rel_path)
            os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
            jobs_by_file.setdefault(rel_path, []).append(Job(src_abs, dst_abs, new_size, d_cfg.get('priority', 99), dest_uuid=d_cfg['uuid']))
            total_job_bytes += new_size

    if not jobs_by_file:
        print("No jobs created.")
        unmount_all(not args.umount)
        return
    
    final_jobs_list = []
    sorted_file_keys = sorted(jobs_by_file.keys())

    for rel_path in sorted_file_keys:
        jobs_for_file = sorted(jobs_by_file[rel_path], key=lambda j: j.priority)
        final_jobs_list.extend(jobs_for_file)

    print(f"Starting Copy. Total data: {format_bytes(total_job_bytes)}")
    time.sleep(1)
    
    display_loop(final_jobs_list, config.get('max_parallel_processes', 2), total_job_bytes)
    
    end_time = time.time()
    end_time_str = format_time_detailed(end_time - start_time)

    print(f"\nCopy operation finished after {end_time_str}.")
    
    if args.umount:
        unmount_all()
    elif AUTO_MOUNTED_PATHS:
        unmount_all(prompt=True)

if __name__ == "__main__":
    main()