i = 0;
temp=$(mktemp -p .); for file in Vivia*
do 
mv "$file" $temp;
mv $temp $(printf "Vivian_%0.1d.png" $i)
i=$((i+1))
done
