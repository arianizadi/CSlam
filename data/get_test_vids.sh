#!/bin/bash
yt-dlp --split-chapters -f 137 -o "%(section_title)s.%(ext)s" "https://www.youtube.com/watch?v=yIzn6Q-eku8"
ls "Rain Walk in Amsterdam・4K ASMR"* | sort | awk '{printf "mv \"%s\" %03d.mp4\n", $0, NR-1}' | bash