#!/bin/bash

# Execute retrieval_stat_overview_plots.py for several test cases.
l_lim=000
u_lim=024
script_name=retrieval_stat_overview_plots.py

for ii in $(seq -w $l_lim $u_lim);
    do
        python3 $script_name "$ii"
        wait
    done

exit
