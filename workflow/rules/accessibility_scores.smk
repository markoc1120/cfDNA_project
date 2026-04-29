# LWPS calculation
rule calculate_lwps:
    input:
        matrix=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}.npy",
        config="confs/thesis.yaml"
    output:
        temp(f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}_lwps.npy")
    params:
        statistic="lwps"
    resources:
        runtime=10
    group: "lwps"
    script:
        "../scripts/calculate_statistics.py"

# FDI calculation
rule calculate_fdi:
    input:
        matrix=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}.npy",
        config="confs/thesis.yaml"
    output:
        temp(f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}_fdi.npz")
    params:
        statistic="fdi"
    resources:
        runtime=10
    group: "fdi"
    script:
        "../scripts/calculate_statistics.py"

# IFS calculation
rule calculate_ifs:
    input:
        matrix=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}.npy",
        config="confs/thesis.yaml"
    output:
        temp(f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}_ifs.npz")
    params:
        statistic="ifs"
    resources:
        runtime=10
    group: "ifs"
    script:
        "../scripts/calculate_statistics.py"

# PFE calculation
rule calculate_pfe:
    input:
        matrix=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}.npy",
        config="confs/thesis.yaml"
    output:
        temp(f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}_pfe.npz")
    params:
        statistic="pfe"
    resources:
        runtime=10
    group: "pfe"
    script:
        "../scripts/calculate_statistics.py"

# OCF calculation
rule calculate_ocf:
    input:
        matrix=f"{INFERENCE_OUTPUT_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}.npy",
        config="confs/thesis.yaml"
    output:
        temp(f"{ACCESSIBILITY_DIR}{{sample}}__{{dhs_file}}_{COVERAGE_SUFFIX}_ocf.npy")
    params:
        statistic="ocf"
    resources:
        runtime=10
    group: "ocf"
    script:
        "../scripts/calculate_statistics.py"
