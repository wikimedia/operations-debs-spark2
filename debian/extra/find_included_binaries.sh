#!/bin/bash

# Include all jars except for hadoop-* jars.
find jars -type f -not -wholename 'jars/hadoop-*.jar' -exec file {} \; | grep -v text | awk -F ':' '{print $1}'

# Also include all binaries in these directories.
find {python/lib,yarn,R,debian/extra} -type f -exec file {} \; | grep -v text | awk -F ':' '{print $1}'
