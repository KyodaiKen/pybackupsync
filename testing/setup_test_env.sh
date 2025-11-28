#!/bin/bash

# Configuration
TEST_DIR="./pybackup_test_env"
MNT_TMP="/mnt/tmp_pybackup_setup"

# Clean up previous runs
echo "--- Cleaning up previous test environment ---"
sudo umount $MNT_TMP 2>/dev/null
# Detach all loop devices associated with our test files
for img in "$TEST_DIR"/*.img; do
    if [ -f "$img" ]; then
        sudo losetup -j "$img" | cut -d: -f1 | xargs -r sudo losetup -d
    fi
done
sudo rm -rf $TEST_DIR
sudo rm -rf $MNT_TMP
rm -f config_test.yaml

# Create directories
mkdir -p $TEST_DIR
sudo mkdir -p $MNT_TMP

echo "--- Creating Virtual Drives ---"
# 1. Source Drive (100MB)
fallocate -l 100M $TEST_DIR/source.img
# 2. Dest Drive 1 (60MB - Small to force deletion)
fallocate -l 60M $TEST_DIR/dest1.img
# 3. Dest Drive 2 (60MB)
fallocate -l 60M $TEST_DIR/dest2.img

# Attach to loop devices
echo "--- Attaching to Loopback Devices ---"
LOOP_SRC=$(sudo losetup -f --show $TEST_DIR/source.img)
LOOP_DST1=$(sudo losetup -f --show $TEST_DIR/dest1.img)
LOOP_DST2=$(sudo losetup -f --show $TEST_DIR/dest2.img)

# Format drives
echo "--- Formatting Drives (exfat) ---"
sudo mkfs.exfat -q $LOOP_SRC
sudo mkfs.exfat -q $LOOP_DST1
sudo mkfs.exfat -q $LOOP_DST2

# Get UUIDs
UUID_SRC=$(sudo blkid -s UUID -o value $LOOP_SRC)
UUID_DST1=$(sudo blkid -s UUID -o value $LOOP_DST1)
UUID_DST2=$(sudo blkid -s UUID -o value $LOOP_DST2)

echo "UUID Source: $UUID_SRC"
echo "UUID Dest1:  $UUID_DST1"
echo "UUID Dest2:  $UUID_DST2"

# --- POPULATE DESTINATION 1 (Force Deletion Scenario) ---
echo "--- Populating Destination 1 (Simulating Old Backups) ---"
sudo mount $LOOP_DST1 $MNT_TMP
sudo mkdir -p $MNT_TMP/backups/primary
# Create 4 files of 10MB each (40MB total used out of 60MB)
# Timestamps are important for "oldest" policy
sudo dd if=/dev/zero of=$MNT_TMP/backups/primary/old_file_1.dat bs=1M count=10 status=none
sudo touch -d "2023-01-01" $MNT_TMP/backups/primary/old_file_1.dat

sudo dd if=/dev/zero of=$MNT_TMP/backups/primary/old_file_2.dat bs=1M count=10 status=none
sudo touch -d "2023-01-11" $MNT_TMP/backups/primary/old_file_2.dat

sudo dd if=/dev/zero of=$MNT_TMP/backups/primary/old_file_3.dat bs=1M count=10 status=none
sudo touch -d "2023-02-01" $MNT_TMP/backups/primary/old_file_3.dat

sudo dd if=/dev/zero of=$MNT_TMP/backups/primary/old_file_4.dat bs=1M count=5 status=none
sudo touch -d "2023-02-22" $MNT_TMP/backups/primary/old_file_4.dat

echo "   -> Dest1 State: 40MB Used, ~15MB Free. Contains files from Jan, Feb, Mar, Apr."
sudo umount $MNT_TMP

# --- POPULATE SOURCE ---
echo "--- Populating Source Drive ---"
sudo mount $LOOP_SRC $MNT_TMP
sudo mkdir -p $MNT_TMP/data/photos
# Create a NEW 25MB file
# 25MB is larger than the 15MB free space on Dest1
sudo dd if=/dev/urandom of=$MNT_TMP/data/photos/new_vacation_video.mp4 bs=1M count=11 status=none
echo "   -> Source State: Created 11MB file 'new_vacation_video.mp4'"
sudo umount $MNT_TMP

# --- GENERATE CONFIG YAML ---
cat <<EOF > config_test.yaml
max_parallel_processes: 2

source:
  uuid: "$UUID_SRC"
  mount_point: "$TEST_DIR/mnt_src"
  path: "data/photos"
  exclude_patterns: []

destinations:
  - uuid: "$UUID_DST1"
    mount_point: "$TEST_DIR/mnt_dst1"
    path: "backups/primary"
    priority: 1
    deletion_policy: "oldest"

  - uuid: "$UUID_DST2"
    mount_point: "$TEST_DIR/mnt_dst2"
    path: "archive/secondary"
    priority: 2
    deletion_policy: "month_dupes"
EOF

echo "--- Setup Complete ---"
echo "1. Config file created at: ./config_test.yaml"
echo "2. Run the backup with:"
echo "   sudo python3 pybackupsync.py -C config_test.yaml -um"
