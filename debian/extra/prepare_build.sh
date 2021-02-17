set -e

extra_dir=$(dirname $(realpath $0))

# Some PySpark dependencies are
# binary C extensions, and as such need to be installed for the Python versions
# that PySpark will be run with.  These may not be the same as are installed via
# Debian.  This command will download the Python dependencies as wheel files, which
# will be unzipped during installation.  The PYTHONPATH will be set approriately
# to load these files at runtime in spark-env.sh.
${extra_dir}/download_python_dependencies.sh

# download_hadoop_dependencies.sh will download different versions of Hadoop dependency
# jars into debian/extra/hadoop.  This is only necessary if the version of Hadoop in your cluster
# is different than the one provided in the Spark dist.  Read the docs of that script or more info.
# If debian/extra/hadoop exists during the build,

if [ -n "${WMF_HADOOP_VERSION}" ]; then
    ${extra_dir}/download_hadoop_dependencies.sh ${WMF_HADOOP_VERSION}
fi

${extra_dir}/find_included_binaries.sh
