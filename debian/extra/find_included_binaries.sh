#!/bin/bash


# Once you've imported the Spark 2 tarball and downloaded dependencies, you need to explicitly declare all
# binary files that need to be included in the package.

if [ -d debian/extra/hadoop ]; then
    (
        find {python/lib,yarn,R,debian/extra} -type f -exec file {} \; | grep -v text | awk -F ':' '{print $1}' && \
        find jars -type f -not -wholename 'jars/hadoop-*.jar' -exec file {} \; | grep -v text | awk -F ':' '{print $1}'
    ) | sort > debian/source/include-binaries
else
    find {jars,python/lib,yarn,R,debian/extra} -type f -exec file {} \; | grep -v text | awk -F ':' '{print $1}' | sort > debian/source/include-binaries
fi


# # If we are using custom Hadoop version, then skip the ones in jars/hadoop-*.jar
# if [ -d  ]; then
#     find {jars,python/lib,yarn,R,debian/extra} -type f -not -wholename 'jars/hadoop-*.jar' -exec file {} \; | grep -v text | awk -F ':' '{print $1}' | sort > debian/source/include-binaries
# else
#     find {jars,python/lib,yarn,R,debian/extra} -type f -exec file {} \; | grep -v text | awk -F ':' '{print $1}' | sort > debian/source/include-binaries
# fi


