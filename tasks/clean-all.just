# run clean tasks
[group('clean-all')]
clean-all: clean-inputs

# clean all inputs
[group('clean-all')]
clean-inputs:
    rm -f raw_data/training_dhs/*.bed
    rm -f raw_data/training_frags/*.hg38.frag.gz
    rm -f raw_data/inference_dhs/*.bed
    rm -f raw_data/training_dhs/.negatives_done
    rm -f raw_data/inference_frags/*.hg38.frag.gz
    rm -rf raw_data/training_dhs_sorted/
    rm -rf raw_data/training_frags_sorted/
    rm -rf raw_data/inference_dhs_sorted/
    rm -f raw_data/meta_data.csv
    rm -rf data/
