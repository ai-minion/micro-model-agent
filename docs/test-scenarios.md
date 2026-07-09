# Test Scenarios for Micro Model Agent

## Scenario 1: Finding All Python Classes Using repo.search
**Description:** Find all Python files that define a class.
**Prompt:** List all Python classes defined in the repository.
**Tool Exercised:** repo.search

## Scenario 2: Reading and Summarizing a Specific Source File Using repo.read
**Description:** Read and summarize a specific source file.
**Prompt:** Provide a summary of the `repo.search` implementation.
**Tool Exercised:** repo.read

## Scenario 3: Creating a New Utility Module Using repo.write_files
**Description:** Create a new utility module from scratch.
**Prompt:** Write a utility function to calculate the Fibonacci sequence.
**Tool Exercised:** repo.write_files

## Scenario 4: Making a Targeted Single-Line Edit Using repo.write_patch
**Description:** Make a targeted single-line edit to an existing file.
**Prompt:** Add a docstring to the `repo.search` function.
**Tool Exercised:** repo.write_patch

## Scenario 5: Running the Test Suite Using test.run
**Description:** Run the test suite and report results.
**Prompt:** Execute the test suite and provide the results.
**Tool Exercised:** test.run