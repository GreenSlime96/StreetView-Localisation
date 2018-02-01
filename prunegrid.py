import argparse
import sys

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('api', type=str, help='Google Maps API key')
    args = parser.parse_args()

    r_session = requests.Session()
    base_URI = 'https://maps.googleapis.com/maps/api/streetview/metadata?' \
        + 'location={}&key=' + args.api

    points = set()
    for line in sys.stdin.read().splitlines():
        response = r_session.get(base_URI.format(line))
        response = response.json()

        if response['status'] != 'OK':
            continue

        if "Google" in response['copyright']:
            location = response['location']
            coords = (location['lat'], location['lng'])

            if coords not in points:
                print("{},{}".format(coords[0], coords[1]))
                points.add(coords)


if __name__ == "__main__":
    main()
