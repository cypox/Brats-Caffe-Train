#include <gflags/gflags.h>
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <caffe/caffe.hpp>
#include <caffe/sgd_solvers.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "database.hpp"
#include "classify.hpp"


int main(int argc, char** argv)
{
    /* Converting mnist database to lmdb
    convert_dataset("mnist/train-images-idx3-ubyte",
                    "mnist/train-labels-idx1-ubyte",
                    "lmdb-mnist", "lmdb");
    //*/

    /* Solving from param
    std::shared_ptr< caffe::SGDSolver<double> > solver;
    caffe::SolverParameter param;
    /* TODO : FILL PARAM HERE
    solver.reset(new caffe::SGDSolver<double>(param));
    solver->Solve();
    //*/

    /* Solving from file
    std::shared_ptr< caffe::SGDSolver<double> > solver;
    solver.reset(new caffe::SGDSolver<double>("network/solver.prototxt"));
    solver->Solve();
    //*/

    //* Testing the network obtained from file
    Classifier classifier("network/solver.prototxt", "network/trained.caffemodel",
                          "mean_file", "label_file");

    string file = "test-image.jpg";
    cv::Mat img = cv::imread(file, -1);
    CHECK(!img.empty()) << "Unable to decode image " << file;
    std::vector<Prediction> predictions = classifier.Classify(img);

    /* Print the top N predictions. */
    for (size_t i = 0; i < predictions.size(); ++i) {
        Prediction p = predictions[i];
        std::cout << std::fixed << std::setprecision(4) << p.second << " - \""
                  << p.first << "\"" << std::endl;
    }
    //*/

    return 0;
}
