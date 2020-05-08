#!/usr/bin/env bash

URL="https://gitlab.ika.rwth-aachen.de/cam2bev/cam2bev-data/-/archive/master/cam2bev-data-master.tar.gz"
FILE="cam2bev-data.tar.gz"

set -e
cd $(dirname $0)

echo "Downloading Cam2BEV Data from $URL to $(realpath $FILE) ..."
wget -q --show-progress -c -O $FILE $URL

echo -n "Extracting $FILE to $(pwd) ... "
tar -xzf $FILE
rm $FILE
mv cam2bev-data-master/* .
echo "done"

for f in $(find . -name "*.tar.gz")
do
  echo -n "  Extracting $f ... "
  tar -xzf $f -C "$(dirname $f)"
  rm $f
  echo "done"
done

echo "Successfully downloaded Cam2BEV Data to $(pwd)"
