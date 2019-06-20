mkdir data
cd data

dir="http://people.csail.mit.edu/tianxiao/data"

wget $dir/yelp.zip
unzip yelp.zip
rm yelp.zip

wget $dir/yahoo.zip
unzip yahoo.zip
rm yahoo.zip
