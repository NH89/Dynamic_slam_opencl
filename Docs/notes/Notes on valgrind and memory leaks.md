NB the Intel and AMD Opencl ICDs trigger memory leak reports in Valgrind.
(THese _might_ be false positives, and are not part of Dynamic_slam_opemcl.





valgrind --leak-check=full --num-callers=24  --log-file=/home/nick/Programming/OpenCV/MySLAM/valgrind-log-Fri2ndAug2024.txt     ../build/Dynamic_slam  /home/nick/Programming/OpenCV/MySLAM/Dynamic_slam_opencl/src/local_conf/filepaths_ThinkPad.json
