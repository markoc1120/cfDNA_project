# LWPS calculation
rule calculate_lwps:
    input:
        matrix=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy",
        config="confs/thesis.yaml"
    output:
        temp(f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_downsampled_lwps.npy")
    params:
        statistic="lwps"
    resources:
        mem_mb=100,
        runtime=10
    group: "lwps"
    script:
        "../scripts/calculate_statistics.py"

# FDI calculation
rule calculate_fdi:
    input:
        matrix=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy",
        config="confs/thesis.yaml"
    output:
        temp(f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_downsampled_fdi.npz")
    params:
        statistic="fdi"
    resources:
        mem_mb=300,
        runtime=10
    group: "fdi"
    script:
        "../scripts/calculate_statistics.py"

# IFS calculation
rule calculate_ifs:
    input:
        matrix=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy",
        config="confs/thesis.yaml"
    output:
        temp(f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_downsampled_ifs.npz")
    params:
        statistic="ifs"
    resources:
        mem_mb=60,
        runtime=10
    group: "ifs"
    script:
        "../scripts/calculate_statistics.py"

# PFE calculation
rule calculate_pfe:
    input:
        matrix=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy",
        config="confs/thesis.yaml"
    output:
        temp(f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_downsampled_pfe.npz")
    params:
        statistic="pfe"
    resources:
        mem_mb=100,
        runtime=10
    group: "pfe"
    script:
        "../scripts/calculate_statistics.py"

# OCF calculation
rule calculate_ocf:
    input:
        matrix=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_downsampled.npy",
        config="confs/thesis.yaml"
    output:
        temp(f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_downsampled_ocf.npy")
    params:
        statistic="ocf"
    resources:
        mem_mb=160,
        runtime=10
    group: "ocf"
    script:
        "../scripts/calculate_statistics.py"
