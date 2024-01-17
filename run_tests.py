import subprocess

# Function to run a script and save its output to a text file
def run_script(script_name, output_file):
    print(f"Running {script_name}...")
    
    result = subprocess.run(['python', script_name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Save standard output to a text file
    with open(output_file, 'w') as f:
        f.write(result.stdout)
    
    # Print the standard error (if any)
    print(f"Error (if any) for {script_name}:\n")
    print(result.stderr)

# # Run the first script and save output to a text file
# run_script('ensemblenntest.py', 'ensemblenntest_output.txt')

# Run the second script and save output to a text file
run_script('dilatedconvnetwork.py', 'dilatedconvnetwork_output.txt')

# Run the third script and save output to a text file
run_script('cnntest.py', 'cnntest_output.txt')

# Run the fourth script and save output to a text file
run_script('grutest.py', 'grutest_output.txt')

# Run the fifth script and save output to a text file
run_script('rnntest.py', 'rnntest_output.txt')

# Run the sixth script and save output to a text file
run_script('lstmtest.py', 'lstmtest_output.txt')

# Run the seventh script and save output to a text file
run_script('probabilisticnntest.py', 'probabilisticnntest_output.txt')

# Run the eight script and save output to a text file
run_script('resnettest.py', 'resnettest_output.txt')
