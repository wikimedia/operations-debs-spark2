#!/bin/bash



# set the PIP variable if you need to override your path to pip
pip=${PIP:-pip3}

extra_dir=$(dirname $(realpath $0))
base_dir=$(dirname $(dirname $extra_dir))

echo "Downloading extras for Spark 2: pyarrow"
echo "Please make sure that you are running this script on the same build architecture"
echo "on which you will be be building and installing this Spark 2 package."

echo "Removing previously downloaded extras..."
rm -rvf $extra_dir/pyarrow* $extra_dir/python/pyarrow*

# Determine arrow version.
arrow_verison=$(ls $base_dir/jars/arrow-format-*.jar | sed 's/.*-\([0-9\.][0-9\.]*\).jar/\1/')

# Download pyarrow wheel from pypi and unzip it.
echo "Downloading pyarrow==$arrow_version wheel into $extra_dir/python"
$pip download --no-deps -d $extra_dir/python pyarrow==$arrow_verison

echo "Done. Wheels in $extra_dir will be unzipped to /usr/lib/spark2/python on install."
echo "Remember to git add and commit your new extra files."
