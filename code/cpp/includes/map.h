#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>

class Map{

    


    public:
    int size_x;
    int size_y;
    int resolution;
    float offset_x;
    float offset_y;
    int max_x, max_y, min_x, min_y;
    cv::Mat map;
    void read_map(char *mapName, float *map);

};