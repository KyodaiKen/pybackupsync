# PyBackupSync

**PyBackupSync** is a robust, terminal-based parallel backup synchronization tool designed for Linux. It manages the copying of new files from a source drive to multiple destination drives simultaneously, handling automatic mounting, disk space management, and smart file deletion to ensure backups always fit.

## 🚀 Features

* **Parallel Processing:** Copies files to multiple hard drives at the same time using configurable concurrency.
* **Smart Job Scheduling:** Interleaves jobs to ensure multiple threads do not bottleneck on the same destination drive simultaneously.
* **Auto-Mounting:** Automatically detects and mounts source and destination partitions by UUID.
* **Space Management:** Calculates net space requirements and deletes old files on destinations if space is needed.
* **Scoped Safety:** Deletion logic is strictly scoped to the specific destination directories being updated, preventing accidental data loss elsewhere on the drive.
* **SSD Optimization:** Automatically detects if a destination is an SSD and runs `fstrim` after cleanup to maintain drive performance.
* **Rich progress UI:** Features a responsive, bordered terminal interface with individual progress bars per file and a global total progress bar, powered by `pv` (Pipe Viewer).
* **Deletion Policies:** Supports "Oldest" (FIFO) and "Month Duplicates" (Smart thinning) policies.

## 🛠️ Prerequisites

This application requires **Linux** (due to dependencies on `mount`, `lsblk`, `findmnt`, and `fstrim`).

### 1. System Dependencies
You must have `pv` (Pipe Viewer) installed on your system.

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install pv
```

#### Fedora / DNF / YUM

```bash
sudo dnf install pv
```

```bash
sudo yum install pv
```

**Important!** The version of `pv` 1.9.xx or newer is required. Check your version using `pv --version` and upgrade when necessary.
This tool will break if the numeric output feature `-n` in conjunction with all `-F` features are not available!

### 2. Python Dependencies
```bash
pip3 install pyyaml
```

## ⚙️ Configuration
Create one or more configuration file(s). Per default, when none has been given, PyBackupSync looks in `~/.pybackupsync.yaml`.
With the `-C config.yaml` parameter, you can use any other configuration file as if you would use different profiles.

### Example Configuration
```yaml
max_parallel_processes: 2

source:
  uuid: "1234-5678-AAAA-BBBB"       # UUID of the source partition
  mount_point: "/mnt/backup_src"    # Temporary mount point
  path: "data/photos"               # Relative path to backup from
  exclude_patterns:                 # Glob patterns to exclude from source
    - "*.tmp"
    - "cache"
    - ".DS_Store"

destinations:
  - uuid: "9876-5432-CCCC-DDDD"
    mount_point: "/mnt/backup_dest_1"
    path: "backups/primary"         # Files will go to /mnt/backup_dest_1/backups/primary and sub directories
    priority: 1                     # Higher priority (lower number) starts first
    deletion_policy: "month_dupes"  # Policy for making space
    deletion_exclude_patterns: []   # Directories or files to exclude from deletion candidates

  - uuid: "1111-2222-EEEE-FFFF"
    mount_point: "/mnt/backup_dest_2"
    path: "archive/secondary"
    priority: 2
    deletion_policy: "oldest"
    deletion_exclude_patterns: ["*"] # "*" disables deletion entirely for this drive
```

### Deletion Policies

* `oldest`: Deletes the oldest files (based on ctime) until enough space is freed.
* `month_dupes`: Groups files by Year-Month. If a month contains multiple files, it keeps the newest one and marks the rest for deletion, starting from the oldest overall. This is useful for thinning out daily backups into monthly archival snapshots.

### Disabling Deletion
To strictly prevent a specific destination from ever deleting files, set:

```yaml
deletion_exclude_patterns: ["*"]
```

## 🚀 Usage
```bash
sudo python3 pybackupsync.py [OPTIONS]
```

### Command Line Arguments

| Flag | Long Flag | Description |
| :--- | :--- | :--- |
| `-C` | `--config` | Path to a specific YAML config file. If omitted, `~/.pybackupsync.yaml` will be used, if it exists. |
| `-um` | `--umount` | Unmount all automatically mounted drives after successful completion. |
| `-s` | `--slow` | **Debug:** Limits copy speed to 1MB/s to test the UI and threading. |

### Example

```bash
# Run with a custom config and auto-unmount on success
sudo python3 pybackupsync.py -C ./my_backup_config.yaml -um
```

## ⚠️ Safety & Logic

* **Net Space Calculation:** The tool calculates the *difference* between the new file size and the existing file size (if overwriting). It only triggers deletion if the *net* change requires more space than is currently free.
* **Scope:** When deleting files to make space, the tool **only scans the specific directories** that are receiving new files. It will not touch other folders on the destination drive.
* **Interleaving:** Jobs for the same destination UUID are run sequentially, but jobs for *different* UUIDs can run in parallel, preventing one fast drive from hogging all worker threads while another is busy.

## 📝 License

[MIT License](LICENSE)
