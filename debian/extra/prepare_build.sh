set -e

extra_dir=$(dirname $(realpath $0))

# Some PySpark dependencies are
# binary C extensions, and as such need to be installed for the Python versions
# that PySpark will be run with.  These may not be the same as are installed via
# Debian.  This command will download the Python dependencies as wheel files, which
# will be unzipped during installation.  The PYTHONPATH will be set approriately
# to load these files at runtime in spark-env.sh.
${extra_dir}/download_python_dependencies.sh

# Find binary files we're going to include in the .deb and list then in include-binaries.
${extra_dir}/find_included_binaries.sh | sort > debian/source/include-binaries

