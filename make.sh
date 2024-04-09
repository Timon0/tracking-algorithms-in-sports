#!/usr/bin/env bash
python setup.py develop

python libs/DCNv2/setup.py build develop
python SportsTracking/trackers/motr/ops/setup.py build develop