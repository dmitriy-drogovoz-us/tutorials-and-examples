#!/bin/bash

SLEEP_SECONDS
SEARCH_STRING="Ottawdddda"
if bash test_model.sh | grep -q -m 1 "$SEARCH_STRING"; then
  echo "SUCCESS: String '$SEARCH_STRING' found in logs!"
else
  echo "Error"
  exit 1
fi
