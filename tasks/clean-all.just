# run clean tasks
[group('clean-all')]
clean-all: clean-inputs

# clean all inputs
[group('clean-all')]
clean-inputs:
    rm -f raw_data/training_dhs/*.bed
    rm -f raw_data/cristiano_healthy/*.hg38.frag.gz
    rm -f raw_data/inference_dhs/*.bed
    rm -f raw_data/cristiano_cfdnas/*.hg38.frag.gz
    rm -rf raw_data/training_dhs_sorted/
    rm -rf raw_data/cristiano_healthy_sorted/
    rm -rf raw_data/inference_dhs_sorted/
    rm -rf data/training_frags/
    rm -rf data/inference_sorted_frags/
    rm -rf data/inference/
    rm -rf data/final_matrices/
    rm -rf raw_data/training_dhs_sorted/
    rm -rf raw_data/cristiano_healthy_sorted/
    rm -rf raw_data/inference_dhs_sorted/
