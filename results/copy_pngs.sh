find . -type f -name "*.png" -not -path "./pngs/*" -exec bash -c 'cp "$1" "pngs/$(echo "$1" | sed "s|^\./||; s|/|_|g")"' _ {} \;
tar -czf pngs.tar.gz pngs
