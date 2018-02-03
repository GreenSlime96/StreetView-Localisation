#!/bin/bash

# Checks for API key, disregards validity
if [[ $# -eq 0 ]] ; then
  echo 'missing Google Maps API key'
  exit 0
fi

# http://maps.googleapis.com/maps/api/streetview/metadata?size=640x640&heading=180&pitch=0&fov=120&location=52.204446,0.117632&key=AIzaSyCtwK9v361GAoMqNBXAYdVRHptYiV6y4yA
# Test API key
api_check="https://maps.googleapis.com/maps/api/streetview/metadata?"
api_check+="location=0,0&key=${1}"

if [[ $(curl -s $api_check) == *"REQUEST_DENIED"* ]]; then
  echo "invalid API key: ${1}"
  exit
fi

# Base URL including API key
BASE_URI="https://maps.googleapis.com/maps/api/streetview?"
BASE_URI+="key=$1&size=640x640"

# Ensures unique folders
DIR=${2:-$(date +%Y%m%d_%H%M%S)}

if [ -d "$DIR" ]; then
  echo "folder ${DIR} exists, please delete"
  exit
else
  echo "saving images in ${DIR}"
  mkdir "$DIR"
fi

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
    curl -s -o "${DIR}/${counter}_${num}.jpg" "${uri}"
  done

  curl -s -o "${DIR}/${counter}_4.jpg" "${uri}&pitch=90"

  counter=$((counter+1))
done < "/dev/stdin"

echo "finished processing ${counter} locations"