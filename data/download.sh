# download all datas
echo "\033[32;40mdownloading datasets, the time cunsumption depends on your web speed(several minutes to tens of hours)...\033[0m"
wget -c http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-00 &
wget -c http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-01 1>/dev/null 2>&1 &
wget -c http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-02 1>/dev/null 2>&1 &
wget -c http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-03 1>/dev/null 2>&1 &
wget -c http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-04 1>/dev/null 2>&1 &
wget -c http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-05 1>/dev/null 2>&1 & 
# wget http://ec2-52-11-11-89.us-west-2.compute.amazonaws.com/data/challenge16/features/c3d/activitynet_v1-3.part-05 & 

# wait %0
# wait %1
# wait %2
# wait %3
# wait %4
# wait %5
wait 
# merge them into a single hdf5 file
echo "\033[32;40mmerge datasets into a single hdf5 file(minutes)\033[0m"
cat activitynet_v1-3.part* > anet_v1.3.c3d.zip
rm activitynet_v1-3.part*
unzip anet_v1.3.c3d.zip
mv sub*.hdf5 anet_v1.3.c3d.hdf5
rm anet_v1.3.c3d.zip

# download the dense captioning data
echo "\033[32;40mdownload dense captioning dataset(minutes)...\033[0m"
wget https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip
unzip captions.zip -d densecap 
rm captions.zip

echo DONE!

