# Street View Video Localisation

This project aims to provide an easy way to localise video using a dataset of Google Street View Images.

## Requirements

`pip install geopy, requests`

## Components

The scraper is split into three components: grid-generator (makegrid), grid-pruner (prunegrid) and scraper (scraper).

### Grid Generator

The grid-generator takes as input the longitude and latitude of a starting center location, and optional inputs radius of the circle (default: 10) and distance between points in meters (default: 5). The results are printed to stdout, where they should be piped to the pruner or stored in a file.

### Grid Pruner

The grid-generator makes no assumptions about the distribution of coordinates; it simply uniformly distributes points along the surface in a circle of equal distances apart. The objective of the pruner then is to eliminate invalid coordinates, and it does this by consulting the Google Street View metadata API for the nearest location corresponding to the given coordinate. It also filters the results such that only Google-captured images are used, which are expected to be consistent.

This takes as input a (through standard input) the coordinates separated by a newline, and the Google Maps API key as an argument. Again the output should be piped or saved to a file.

### Scraper

The scraper reads from standard input a list of pruned coordinates and takes as argument the Google Maps API key. Given a coordinate, the largest photo size (640x640) is retrieved and named such that the first number corresponds to the entry in standard input, and the second number from 0 to 3 corresponds to the bearing (where 0 is North, 90 is East, etc). Additionally the scraper includes the top view, marked by the bearing-id 4.

The images are saved in a directory named after the time the script ran.




