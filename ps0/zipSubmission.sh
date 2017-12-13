#!/usr/bin/env bash

# Compress the current problem set directory into a zip 
# archive for submission via bCourses
#
# Usage:
# > ./zipSubmission.sh
#
# This will produce the archive "submission.zip" in the problem 
# set directory.

PSNAME=${PWD##*/}
CURDIR=$(pwd)

cd ~/problem-sets/
zip -r submission.zip $PSNAME
mv submission.zip ~/problem-sets/$PSNAME
cd $CURDIR