"""Download and condense the public data of the global estimate (run where the internet is reachable).

World Bank, World Development Indicators (API v2): EG.ELC.ACCS.RU.ZS (access to electricity, rural, % of rural
population) and SP.RUR.TOTL (rural population), latest year from 2015 to 2024, economies with incomplete rural
access, excluding aggregates.
FAO, "Production: Crops and livestock products", items rice (27) and maize (56), element production (5510), as
served by Our World in Data (FAOSTAT's own API was unavailable when the data were retrieved on 14 October 2026);
mean of 2022-2024.

usage: python fetch_global_data.py <out_dir>
"""
import csv
import io
import json
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

WB = "https://api.worldbank.org/v2"
OWID = "https://ourworldindata.org/grapher/{}.csv?v=1&csvType=full&useColumnShortNames=true"


def get(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read().decode("utf-8")


def world_bank(out):
    countries = json.loads(get(f"{WB}/country?format=json&per_page=400"))[1]
    real = {c["id"]: c for c in countries if c["region"]["value"] != "Aggregates"}

    def indicator(code):
        rows = json.loads(get(f"{WB}/country/all/indicator/{code}?format=json&per_page=20000&date=2015:2024"))[1]
        data = defaultdict(dict)
        for r in rows:
            if r["value"] is not None and r["countryiso3code"] in real:
                data[r["countryiso3code"]][int(r["date"])] = r["value"]
        return data

    access, rural = indicator("EG.ELC.ACCS.RU.ZS"), indicator("SP.RUR.TOTL")
    with open(out / "worldbank_rural_access.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iso3", "year", "rural_access_pct", "rural_pop", "region", "income"])
        for iso in sorted(real):
            if iso not in access or iso not in rural:
                continue
            year = max(access[iso])
            value = access[iso][year]
            if value >= 99.95:
                continue
            nearest = min(rural[iso], key=lambda y: abs(y - year))
            w.writerow([iso, year, f"{value:.2f}", round(rural[iso][nearest]),
                        real[iso]["region"]["id"], real[iso]["incomeLevel"]["id"]])


def fao_via_owid(out):
    means = {}
    for crop, name in (("rice", "rice-production"), ("maize", "maize-production")):
        values = defaultdict(list)
        reader = csv.reader(io.StringIO(get(OWID.format(name))))
        next(reader)
        for row in reader:
            code, year, value = row[-3], int(row[-2]), float(row[-1])
            if len(code) == 3 and code.isalpha() and code.isupper() and 2022 <= year <= 2024:
                values[code].append(value)
        means[crop] = values
    codes = sorted(set(means["rice"]) | set(means["maize"]))
    with open(out / "faostat_production_2022_2024.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["iso3", "rice_t", "maize_t", "n_rice", "n_maize"])
        for c in codes:
            r, m = means["rice"].get(c, []), means["maize"].get(c, [])
            w.writerow([c, round(sum(r) / len(r)) if r else 0, round(sum(m) / len(m)) if m else 0, len(r), len(m)])


if __name__ == "__main__":
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)
    world_bank(target)
    fao_via_owid(target)
    print("written to", target)
