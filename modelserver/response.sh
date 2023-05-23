#!/bin/bash

# Adjust as needed.
TEXTFILE_COLLECTOR_DIR=/root/server/
# Note the start time of the script.
START="$(date +%s)"
A=3.5
TIMET=$(bc <<< "scale=3;$A*($RANDOM/32767)")
# Your code goes here.
sleep $TIMET
# Write out metrics to a temporary file.
END="$(date +%s)"
TOTAL=$(bc <<< "scale=3;$TIMET*1000")
echo $TOTAL
cat << EOF > "$TEXTFILE_COLLECTOR_DIR/myscript.prom.$$"
myscript_duration_miliseconds $TOTAL
myscript_last_run_seconds $END
EOF

# Rename the temporary file atomically.
# This avoids the node exporter seeing half a file.
mv "$TEXTFILE_COLLECTOR_DIR/myscript.prom.$$" \
  "$TEXTFILE_COLLECTOR_DIR/myscript.prom"
