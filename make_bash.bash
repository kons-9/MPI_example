#!/bin/bash

sed s/TEMPLATE/$1/g template.bash > $1.bash
chmod 710 $1.bash

