# run clean tasks
[group('gen-test')]
gen-test: generate

# clean all inputs
[group('gen-test')]
generate:
    python gen_test_data.py
