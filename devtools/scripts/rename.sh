#!/usr/bin/env bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <from-native-to> <from-nonbonded-to>"
    exit 1
fi

uc () { echo $1 | tr '[:lower:]' '[:upper:]'; }  # upper case
lc () { echo $1 | tr '[:upper:]' '[:lower:]'; }  # lower case
tc () { LC=$(lc $1); echo ${LC^}; }              # title case

UC=$(uc $1$2)
LC=$(lc $1$2)
CC=$(tc $1)$(tc $2)

# Rename files
for file in $(git ls-files | grep "NativeNonbonded"); do
    git mv $file ${file/NativeNonbonded/$CC}
done

for file in $(git ls-files | grep "nativenonbonded"); do
    git mv $file ${file/nativenonbonded/$LC}
done

# Replace text in code
git grep -l "NATIVENONBONDED" | xargs sed -i "s/NATIVENONBONDED/$UC/g"
git grep -l "NativeNonbonded" | xargs sed -i "s/NativeNonbonded/$CC/g"
git grep -l "nativenonbonded" | xargs sed -i "s/nativenonbonded/$LC/g"
