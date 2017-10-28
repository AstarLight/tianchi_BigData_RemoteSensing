#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


int main(int argc,char* argv[])
{
    string name;

    if (argc != 2) {
        cout << "Wrong Arguments. Correct use:" << argv[0] << "  [2015|2017]" << endl;
        return -1;
    }

    name = argv[1];

    FILE* fp = fopen(name.c_str(), "r");
    if (!fp)
    {
        printf("fail to open %srect.txt\n", name.c_str());
        return -1;
    }

    while (!feof(fp))
    {

        Rect bouding;
        string image_name;
        float scorce;
        printf("processing %s\n", image_name.c_str());
        fscanf(fp, "%s %d %d %d %d %f", image_name.c_str(), &bouding.x, &bouding.y,
            &bouding.width, &bouding.height, &scorce);

        string image_src_path = name + "/" + image_name;
        Mat src = imread(image_src_path);

        rectangle(src, bouding, Scalar(0, 255, 0), 2);
        string output_path = "output/rectangle/" + image_src_path;
        imwrite(output_path, src);
    }


    return 0;
}