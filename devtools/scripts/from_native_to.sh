#!/usr/bin/env bash

uc=$(echo $1 | tr '[:lower:]' '[:upper:]')  # upper case
lc=$(echo $1 | tr '[:upper:]' '[:lower:]')  # lower case
tc=$(echo ${lc^})                           # title case

# Rename files
for file in $(git ls-files | grep "Native"); do
    git mv $file ${file/Native/$tc}
done

for file in $(git ls-files | grep "native"); do
    git mv $file ${file/native/$lc}
done

# Replace text in code
git grep -l "NATIVE" | xargs sed -i "s/NATIVE/$uc/g"
git grep -l "Native" | xargs sed -i "s/Native/$tc/g"
git grep -l "native" | xargs sed -i "s/native/$lc/g"
