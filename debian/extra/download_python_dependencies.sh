#!/bin/bash

# We want to ship all pyspark dependencies with the spark package itself.
# There are version discrepencies and between versions of python, spark and
# debian that are difficult to resolve.  The easiest thing to do is just
# ship any of the relevant python depdendencies with the spark2 package
# in /usr/lib/spark/python.  However, since some of these dependencies,
# e.g. numpy, pandas and pyarrow are C extensions, they are built for
# specific versions of Python.  These binary dependencies will
# be provided in /usr/lib/spark/pythonX.X, which will be added to
# the PYTHONPATH at runtime in spark-env.sh.


extra_dir=$(dirname $(realpath $0))
debian_dir=$(dirname ${extra_dir})
base_dir=$(dirname ${debian_dir})

# default spark_version to version of spark-core .jar
spark_version=${SPARK_VERSION:-$(ls $base_dir/jars/spark-core*.jar | sed 's/.*-\([0-9\.][0-9\.]*\).jar/\1/')}

# python versions to package dependencies for.
# If you change this list, you must also modify the commands
# to extract python binary dependencies in debian/rules,
# as well as ensure that the list of installed dirs in debian/spark2.dirs
# includes the pythonX.X directories.
python_versions="3.5 3.7"

# Ensure python versions and virtualenv is installed.

if [ -z $(which virtualenv) ]; then
    echo "virtualenv is not installed, aborting."
    exit 1
fi


for python_version in ${python_versions}; do
    python_bin="python${python_version}"
    if [ -z $(which $python_bin) ]; then
        echo "Python version ${python_version} is not installed, aborting."
        exit 1
    fi
done

echo "Downloading python dependencies for PySpark 2..."

echo "Removing previously downloaded python wheel dependencies..."
rm -rvf $extra_dir/python*

for python_version in ${python_versions}; do
    python_bin="python${python_version}"

    # create a virtualenv for this python version
    venv=../venv-build-spark2-${python_bin}
    test -d ${venv} || virtualenv --python=${python_bin} ${venv}

    # non binary python packages will go here
    mkdir -p ${extra_dir}/python
    # binary python packages for specific python version will go here
    mkdir -p ${extra_dir}/${python_bin}

    # use the virtualenv's python pip to create wheel files
    wheel_dir=${extra_dir}/${python_bin}
    echo "Downloading pyspark[sql]==${spark_version} and dependency wheels into ${wheel_dir}"
    set -x
    ${venv}/bin/pip wheel --wheel-dir ${wheel_dir} pyspark[sql]==${spark_version}
    set +x

    # Now that we've got all wheels, remove what we don't need: pyspark itself, since that)
    # is already included in this spark2 package.
    rm ${wheel_dir}/pyspark*.whl
    # And move any non C/binary wheels to regular python/ dir.
    # These might get downloaded and replaced by the next iteration of this
    # loop, but that should be ok, as the python version won't matter.
    echo "Moving non binary pyspark[sql]==${spark_version} dependencies into ${extra_dir}/python"
    mv ${wheel_dir}/*none-any.whl ${extra_dir}/python/
done

echo "Done. Wheels in $extra_dir/python* will be unzipped to /usr/lib/spark2/python or /usr/lib/spark2/pythonX.X on install."
echo "Remember to git add and commit your newly downloaded wheel files."

