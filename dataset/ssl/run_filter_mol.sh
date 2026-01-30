#!/bin/bash
# ============================================================================
# Script to run filter_molecules.py iteratively for multiple input directories
# 
# Usage:
#   1. Get interactive allocation first:
#      salloc -N 1 -C cpu -q interactive -t 04:00:00 -A m3342
#   2. Run this script:
#      bash run_filter_mol.sh
# ============================================================================

# Set paths
BASE_INPUT_DIR="/global/cfs/cdirs/m3342/jhxie/database/pubchem/outputs/stage0_5_parent_ha50_neutral_elem_bins_sdf"
OUTPUT_DIR="/pscratch/sd/y/yeming/AI4M/prediction/filtered_csv"
FILTER_SCRIPT="/pscratch/sd/y/yeming/AI4M/prediction/dataset/ssl/filter_molecules.py"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# Parameters
WORKERS=128
DEBUG_FIRST_SHARD=true  # Set to true to print first molecule's properties for debugging

# Compound range parameters
START_COMPOUND=1
END_COMPOUND=177000000
STEP=500000

# Calculate total shards
TOTAL_SHARDS=$(( (END_COMPOUND - START_COMPOUND + STEP) / STEP ))
CURRENT_SHARD=0

echo "Starting batch processing..."
echo "Total shards to process: ${TOTAL_SHARDS}"

# Iterate through all compound ranges
for (( RANGE_START=${START_COMPOUND}; RANGE_START<${END_COMPOUND}; RANGE_START+=${STEP} )); do
    RANGE_END=$((RANGE_START + STEP - 1))
    
    # Format with leading zeros (9 digits)
    RANGE_START_FMT=$(printf "%09d" ${RANGE_START})
    RANGE_END_FMT=$(printf "%09d" ${RANGE_END})
    
    # Compound range string
    COMPOUND_RANGE="${RANGE_START_FMT}_${RANGE_END_FMT}"
    
    # Input directory
    INPUT_DIR="${BASE_INPUT_DIR}/shard__Compound_${COMPOUND_RANGE}"
    
    # Output file (CSV instead of SDF)
    OUTPUT_FILE="${OUTPUT_DIR}/${COMPOUND_RANGE}.csv"
    
    # Update progress counter
    CURRENT_SHARD=$((CURRENT_SHARD + 1))
    
    echo ""
    echo "============================================="
    echo "Shard ${CURRENT_SHARD}/${TOTAL_SHARDS}: ${COMPOUND_RANGE}"
    echo "============================================="
    
    # Check if input directory exists
    if [ ! -d "${INPUT_DIR}" ]; then
        echo "Skipping: input directory not found"
        continue
    fi
    
    # Check if output file already exists
    if [ -f "${OUTPUT_FILE}" ]; then
        echo "Skipping: output file already exists"
        continue
    fi
    
    # Modify FILE_SUFFIX in filter_molecules.py
    echo "Modifying filter_molecules.py for compound range: ${COMPOUND_RANGE}"
    sed -i "s/FILE_SUFFIX = \"__Compound_.*\.sdf\.gz\"/FILE_SUFFIX = \"__Compound_${COMPOUND_RANGE}.sdf.gz\"/" ${FILTER_SCRIPT}
    
    # Run filter_molecules.py with srun
    echo "Running filter_molecules.py..."
    
    # Add --debug flag for first shard if DEBUG_FIRST_SHARD is true
    DEBUG_FLAG=""
    if [ "${DEBUG_FIRST_SHARD}" = "true" ] && [ ${CURRENT_SHARD} -eq 1 ]; then
        DEBUG_FLAG="--debug"
        echo "DEBUG MODE: Will print first molecule's properties"
    fi
    
    srun -n 1 -c ${WORKERS} python ${FILTER_SCRIPT} \
        --input_dir ${INPUT_DIR} \
        --output ${OUTPUT_FILE} \
        --workers ${WORKERS} \
        ${DEBUG_FLAG}
    
    if [ $? -eq 0 ]; then
        echo "Completed successfully"
    else
        echo "FAILED"
    fi
done

echo ""
echo "============================================="
echo "Batch processing complete!"
echo "Output directory: ${OUTPUT_DIR}"
echo "============================================="
