#!/bin/bash

# Checks for API Key, disregards validity
if [[ $# -eq 0 ]] ; then
  echo 'Error: Missing Google Maps API Key'
  exit 0
fi

# Base URL including API key
BASE_URI="https://maps.googleapis.com/maps/api/streetview?"
BASE_URI+="key=$1&size=640x640"

# Ensures unique folders
DIR=$(date +%Y%m%d_%H%M%S)
mkdir "$DIR"

# Iterate through STDIN input
counter=0
while read line
do
  echo "working on file $counter"
  NEW_URI="${BASE_URI}&location=$line"

  for num in {3..0}
  do
    heading=$((num*90))
    uri="${NEW_URI}&heading=$heading"
    curl -s "${uri}" > "${DIR}/${counter}_${num}.jpg"
  done

  curl -s "${uri}&pitch=90" > "${DIR}/${counter}_4.jpg"

  counter=$((counter+1))
done < "/dev/stdin"
