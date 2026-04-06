# run clean tasks
[group('clean-output')]
clean-output: clean-outputs

# clean all inputs
[group('clean-output')]
clean-outputs:
    rm -rf data/
