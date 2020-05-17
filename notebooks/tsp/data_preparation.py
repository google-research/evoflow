from tqdm.auto import tqdm
from termcolor import cprint
import codecs
from itertools import permutations
from operator import itemgetter
from geopy.distance import distance
import numpy as np
import json
import sys
import random

FNAME_IN = "data/cities15000.txt"
OUT_DIR = "data/"

# see: https://download.geonames.org/export/dump/readme.txt
MAPPING = {
    'name': 1,
    'lat': 4,
    'lon': 5,
    'country_code': 8,
    'population': 14,
    'elevation': 15
}

# used as mapping from code to name and filtering.
COUNTRIES = {
    "FR": "France",
    "DE": "Germany",
    "IT": "Italy",
    "ES": "Spain",
    "PT": "Portugal",
    "BE": "Belgium",
    "NL": "Netherlands",
    "CH": "Switzerland"
}
MIN_LAT = 35  # avoid islands that don't make sense for TSP
BLACKLIST = ['Palma']
TOTAL_CITIES = 24482


def main():

    if len(sys.argv) != 2:
        cprint("Usage %s num_cities" % sys.argv[0], 'red')
        quit()

    num_cities = int(sys.argv[1])

    # city data to array
    cities = []
    f = codecs.open(FNAME_IN, 'r', encoding='utf8')
    pb = tqdm(total=TOTAL_CITIES, desc='Parsing cities', unit='cities')
    for idx, row in enumerate(f):

        row = row.split('\t')

        # keep only cities from the countries we are interested in
        country_code = row[MAPPING['country_code']].upper()
        if country_code in COUNTRIES:
            city = {}
            for field_name, field_idx in MAPPING.items():
                city[field_name] = row[field_idx]
                city['country_name'] = COUNTRIES[country_code]

            if city['population'] == '':
                continue

            if city['name'] in BLACKLIST:
                continue

            city['population'] = int(city['population'])
            city['lat'] = float(city['lat'])
            city['lon'] = float(city['lon'])
            if city['elevation'] != '':
                city['elevation'] = int(city['elevation'])
            else:
                city['elevation'] = 0

            if city['lat'] < MIN_LAT:
                continue
            cities.append(city)

        pb.update()
    pb.close()

    # select cities
    cprint('Found %d cities' % (len(cities)), 'green')
    cprint("Selecting %d largest cities" % num_cities, 'blue')
    cities = sorted(cities, key=lambda k: k['population'], reverse=True)
    cities = cities[:num_cities]

    # assign id
    for idx, city in enumerate(cities):
        city['idx'] = idx

    fname = "%scities_%s.json" % (OUT_DIR, num_cities)
    with open(fname, 'w+') as out:
        out.write(json.dumps(cities))

    # distance matrix
    total = len(cities) * len(cities) - 1
    # Using int16 to save memory as no cities are >65000km appart
    # or the distance computation is wrong as we don't consider Mars cities :)
    distances_matrix = np.zeros((len(cities), len(cities)), dtype='int16')
    pb = tqdm(total=total, desc='Computing geodesic distances')
    for p in permutations(cities, r=2):
        loc1 = (p[0]['lat'], p[0]['lon'])
        loc2 = (p[1]['lat'], p[1]['lon'])
        idx1 = p[0]['idx']
        idx2 = p[1]['idx']

        dist = distance(loc1, loc2).km

        distances_matrix[idx1][idx2] = int(dist)
        pb.update()
    pb.close()

    fname = "%sdistances_%s.npz" % (OUT_DIR, num_cities)
    np.savez_compressed(fname, distances=distances_matrix)
    cprint('data ready! %s' % fname, 'green')


if __name__ == "__main__":
    main()
