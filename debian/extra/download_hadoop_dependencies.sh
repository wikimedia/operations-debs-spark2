#!/bin/bash

# In February 2021, we upgraded our Hadoop cluster to 2.10.  Spark distributions
# do not come pre-packaged for this Hadoop version.  The latest 2.x dist provided
# is for 2.7, which does clashes with the Hadoop 2.10 jars deployed around the cluster.
# To fix, we manually include the Hadoop dependency jars we need in this deb.
# In the future, when we upgrade to Spark 3, we plan to properly build a 'hadoop provided'
# distribution of Spark where it knows how to use Hadoop using Hadoop's installed jars, rather
# than ones shipped with this Spark .deb.
#
# See also https://phabricator.wikimedia.org/T274384

extra_dir=$(dirname $(realpath $0))
debian_dir=$(dirname ${extra_dir})
base_dir=$(dirname ${debian_dir})


hadoop_version="${1}"

if [ -z "${hadoop_version}" ]; then
    echo "Usage: $0 <hadoop-version>"
    exit 1
fi


# Download the Hadoop jars from archiva.wikimedia.org:
hadoop_dependencies="\
hadoop-annotations \
hadoop-auth \
hadoop-client \
hadoop-common \
hadoop-hdfs \
hadoop-mapreduce-client-app \
hadoop-mapreduce-client-common \
hadoop-mapreduce-client-core \
hadoop-mapreduce-client-jobclient \
hadoop-mapreduce-client-shuffle \
hadoop-yarn-api \
hadoop-yarn-client \
hadoop-yarn-common \
hadoop-yarn-server-common \
hadoop-yarn-server-web-proxy"

maven_repo_url="https://archiva.wikimedia.org/repository/mirror-maven-central"

mkdir -p ${extra_dir}/hadoop


for hadoop_dependency in $hadoop_dependencies; do
    jar_url="${maven_repo_url}/org/apache/hadoop/${hadoop_dependency}/${hadoop_version}/${hadoop_dependency}-${hadoop_version}.jar"
    echo "Downloading ${jar_url} in ${extra_dir}/hadoop/"
    wget -P ${extra_dir}/hadoop/ "${jar_url}"
done

# If $extra_dir/hadoop exists, debian/rules will remove the spark provided hadoop-* jars and
# use the ones installed into $extra_dir/hadoop.

