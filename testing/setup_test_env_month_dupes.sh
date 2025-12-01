#!/bin/bash

# --- Configuration ---
TEST_ROOT="test_env_month_dupes"
SOURCE_DIR="${TEST_ROOT}/drive_src/data"
DEST_DIR="${TEST_ROOT}/drive_dest/backups"

# Simulated Drive Paths
SRC_IMAGE="${TEST_ROOT}/src_drive.img"
DEST_IMAGE="${TEST_ROOT}/dest_drive.img"
# We must use unique UUIDs for mounting to work properly
SRC_UUID="a928-8b74-source-1111"
DEST_UUID="b739-1c62-dest-2222"

# File Setup
FILE_SIZE_KB=100
NEEDED_FILES=3 # How many files the new job will add (300KB)

# Calculated Deficit: We will fill the destination drive to force deletion.
# The total space on the DEST_IMAGE will be 1MB.
# 5 existing files (500KB) + 3 new files (300KB) = 800KB needed.
# If we keep the disk small, the free space will be low, forcing cleanup.

# --- Files and Dates (Oldest to Newest) ---
# NOTE: We use ctime (last modification) for deletion logic.
# Month 1 (Oldest): 2 Duplicates (M1_A should be deleted first)
DATE_M1_A=$(date -d "today - 90 days" +%Y%m%d%H%M.%S)
DATE_M1_B=$(date -d "today - 89 days" +%Y%m%d%H%M.%S)

# Month 2 (Middle): 1 Archive (M2 should be deleted second)
DATE_M2=$(date -d "today - 60 days" +%Y%m%d%H%M.%S)

# Month 3 (Newest): 2 Duplicates (M3_A should be deleted last)
DATE_M3_A=$(date -d "today - 30 days" +%Y%m%d%H%M.%S)
DATE_M3_B=$(date -d "today - 29 days" +%Y%m%d%H%M.%S)

# --- Execution ---

echo "--- Setting up PyBackupSync Month Dupes Test Environment ---"

# 1. Cleanup old environment
sudo umount ${TEST_ROOT}/drive_src 2>/dev/null
sudo umount ${TEST_ROOT}/drive_dest 2>/dev/null
rm -rf ${TEST_ROOT}
mkdir -p ${TEST_ROOT}/drive_src
mkdir -p ${TEST_ROOT}/drive_dest

# 2. Create simulated drive images (1MB size)
echo "Creating drive images..."
dd if=/dev/zero of=${SRC_IMAGE} bs=1M count=1 2>/dev/null
dd if=/dev/zero of=${DEST_IMAGE} bs=1M count=1 2>/dev/null

# 3. Format and assign UUIDs (using VFAT for simplicity and UUID support)
echo "Formatting drives..."
sudo mkfs.vfat -F 32 -n SRC_TEST ${SRC_IMAGE} > /dev/null
sudo fatlabel ${SRC_IMAGE} -i ${SRC_UUID} > /dev/null

sudo mkfs.vfat -F 32 -n DEST_TEST ${DEST_IMAGE} > /dev/null
sudo fatlabel ${DEST_IMAGE} -i ${DEST_UUID} > /dev/null

# 4. Mount the images using loop devices
echo "Mounting images to test directories..."
sudo mount -o loop,uid=$(id -u),gid=$(id -g) ${SRC_IMAGE} ${TEST_ROOT}/drive_src
sudo mount -o loop,uid=$(id -u),gid=$(id -g) ${DEST_IMAGE} ${TEST_ROOT}/drive_dest

# 5. Create directories on the new file systems
mkdir -p ${SOURCE_DIR}
mkdir -p ${DEST_DIR}

# 6. Create the base file content (100KB)
echo "Creating 100KB base file..."
dd if=/dev/zero of="${SOURCE_DIR}/file_base.dat" bs=1K count=${FILE_SIZE_KB} 2>/dev/null

# 7. Create existing files in DEST_DIR (5 files total = 500KB)
echo "Creating 5 existing destination files with specific ctimes (500KB total)..."
# M1 (Oldest)
cp "${SOURCE_DIR}/file_base.dat" "${DEST_DIR}/M1_01_oldest_A.dat"
sudo touch -t ${DATE_M1_A} "${DEST_DIR}/M1_01_oldest_A.dat" # TARGET: Delete First

cp "${SOURCE_DIR}/file_base.dat" "${DEST_DIR}/M1_02_keeper_B.dat"
sudo touch -t ${DATE_M1_B} "${DEST_DIR}/M1_02_keeper_B.dat"

# M2 (Middle)
cp "${SOURCE_DIR}/file_base.dat" "${DEST_DIR}/M2_01_archive.dat"
sudo touch -t ${DATE_M2} "${DEST_DIR}/M2_01_archive.dat" # TARGET: Delete Second

# M3 (Newest)
cp "${SOURCE_DIR}/file_base.dat" "${DEST_DIR}/M3_01_oldest_A.dat"
sudo touch -t ${DATE_M3_A} "${DEST_DIR}/M3_01_oldest_A.dat" # TARGET: Delete Third (if needed)

cp "${SOURCE_DIR}/file_base.dat" "${DEST_DIR}/M3_02_keeper_B.dat"
sudo touch -t ${DATE_M3_B} "${DEST_DIR}/M3_02_keeper_B.dat"

# 8. Create new files in SOURCE_DIR (to simulate the job that triggers deletion)
echo "Creating ${NEEDED_FILES} new files in source for copy operation (300KB)..."
for i in $(seq 1 ${NEEDED_FILES}); do
    cp "${SOURCE_DIR}/file_base.dat" "${SOURCE_DIR}/new_file_${i}.dat"
done

# 9. Create Configuration File
echo "Creating test_config.yaml..."
cat << EOF > ${TEST_ROOT}/test_config.yaml
max_parallel_processes: 1

source:
  uuid: "${SRC_UUID}"
  mount_point: "${TEST_ROOT}/drive_src"
  path: "data"
  exclude_patterns: []

destinations:
  - uuid: "${DEST_UUID}"
    mount_point: "${TEST_ROOT}/drive_dest"
    path: "backups"
    priority: 1
    deletion_policy: "month_dupes"
    deletion_exclude_patterns: []
EOF

# 10. Final verification and instructions
echo "--------------------------------------------------------"
echo "Setup Complete."
echo "Source Path: ${TEST_ROOT}/drive_src/data"
echo "Destination Path: ${TEST_ROOT}/drive_dest/backups"
echo "Existing Files (500KB): M1_A (Oldest, Dup), M1_B (Keeper), M2 (Archive), M3_A (Dup), M3_B (Keeper)"
echo "New Files (300KB): 3 new files."
echo "Disk Size: 1MB. Free Space is ~500KB. Need 300KB. No initial cleanup needed."
echo "However, if you set the disk size smaller or add more files, cleanup will be forced."
echo ""
echo "TEST TARGETS (The files that should be deleted if 300KB is needed):"
echo "1. ${DEST_DIR}/M1_01_oldest_A.dat (Oldest Duplicate)"
echo "2. ${DEST_DIR}/M3_01_oldest_A.dat (Newer Duplicate)"
echo "3. ${DEST_DIR}/M2_01_archive.dat (Oldest Single Archive)"
echo "--------------------------------------------------------"
echo "To run the test:"
echo "sudo python3 your_script_name.py -C ${TEST_ROOT}/test_config.yaml"
