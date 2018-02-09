import argparse
import math

import geopy.distance


def generate_hive_locations(current_location, step_distance,
                            step_limit, radius):
    NORTH = 0
    EAST = 90
    SOUTH = 180
    WEST = 270

    xdist = math.sqrt(3) * step_distance  # Distance between column centers.
    ydist = 3 * (step_distance / 2)  # Distance between row centers.

    results = []

    results.append((current_location[0], current_location[1], 0))

    loc = current_location
    ring = 1

    while ring <= radius:

        loc = get_new_coords(loc, ydist * (step_limit - 1), NORTH)
        loc = get_new_coords(loc, xdist * (1.5 * step_limit - 0.5), EAST)
        results.append((loc[0], loc[1], 0))

        for i in range(ring):
            loc = get_new_coords(loc, ydist * step_limit, NORTH)
            loc = get_new_coords(loc, xdist * (1.5 * step_limit - 1), WEST)
            results.append((loc[0], loc[1], 0))

        for i in range(ring):
            loc = get_new_coords(loc, ydist * (step_limit - 1), SOUTH)
            loc = get_new_coords(loc, xdist * (1.5 * step_limit - 0.5), WEST)
            results.append((loc[0], loc[1], 0))

        for i in range(ring):
            loc = get_new_coords(loc, ydist * (2 * step_limit - 1), SOUTH)
            loc = get_new_coords(loc, xdist * 0.5, WEST)
            results.append((loc[0], loc[1], 0))

        for i in range(ring):
            loc = get_new_coords(loc, ydist * (step_limit), SOUTH)
            loc = get_new_coords(loc, xdist * (1.5 * step_limit - 1), EAST)
            results.append((loc[0], loc[1], 0))

        for i in range(ring):
            loc = get_new_coords(loc, ydist * (step_limit - 1), NORTH)
            loc = get_new_coords(loc, xdist * (1.5 * step_limit - 0.5), EAST)
            results.append((loc[0], loc[1], 0))

        # Back to start.
        for i in range(ring - 1):
            loc = get_new_coords(loc, ydist * (2 * step_limit - 1), NORTH)
            loc = get_new_coords(loc, xdist * 0.5, EAST)
            results.append((loc[0], loc[1], 0))

        loc = get_new_coords(loc, ydist * (2 * step_limit - 1), NORTH)
        loc = get_new_coords(loc, xdist * 0.5, EAST)

        ring += 1

    return results


# Returns destination coords given origin coords, distance (Kms) and bearing.
def get_new_coords(init_loc, distance, bearing):
    """
    Given an initial lat/lng, a distance(in kms), and a bearing (degrees),
    this will calculate the resulting lat/lng coordinates.
    """
    origin = geopy.Point(init_loc[0], init_loc[1])
    destination = geopy.distance.distance(meters=distance).destination(
        origin, bearing)
    return (destination.latitude, destination.longitude)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('latitude', type=float, help='starting latitude')
    parser.add_argument('longitude', type=float, help='starting longitude')
    parser.add_argument("-r", "--radius", type=int, default=10,
                        help='radius of hexagon')
    parser.add_argument("-d", "--distance", type=int, default=5,
                        help='separation between points (meters)')

    args = parser.parse_args()

    results = generate_hive_locations((args.latitude, args.longitude),
                                      args.distance, 1, args.radius)

    for point in results:
        print("{},{}".format(point[0], point[1]), flush=True)


if __name__ == "__main__":
    main()
