#!/bin/bash

## for Lesson
echo ${MYSCRIPTS}
cd ${MYSCRIPTS}/../01_class/lesson/kinoco

# git remote set-head origin --auto
git add -A
git commit -m "Commit at $(date "+%Y-%m-%d %T")"
git push origin main # master から main に変更されている
git pull
cd -
