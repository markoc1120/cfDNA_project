# https://just.systems/man/en/

# REQUIRES

find := require("find")
rm := require("rm")
uv := require("uv")

# SETTINGS

set dotenv-load := true

# VARIABLES

PACKAGE := "cfdna"
REPOSITORY := "cfDNA_project"
SOURCES := "src"
WORKFLOW_SCRIPTS := "workflow/scripts"
TESTS := "tests"

# DEFAULTS

# display help information
default:
    @just --list

# IMPORTS

import 'tasks/check.just'
import 'tasks/clean.just'
import 'tasks/clean-all.just'
import 'tasks/gen-test.just'
import 'tasks/commit.just'
import 'tasks/doc.just'
import 'tasks/format.just'
import 'tasks/install.just'
import 'tasks/package.just'
import 'tasks/project.just'
import 'tasks/local-run.just'
import 'tasks/slurm-run.just'
import 'tasks/dry-run.just'
