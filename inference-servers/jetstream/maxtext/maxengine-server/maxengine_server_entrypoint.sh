#!/bin/bash
cd /maxtext
python3 -m MaxText.maxengine_server \
MaxText/configs/base.yml $@