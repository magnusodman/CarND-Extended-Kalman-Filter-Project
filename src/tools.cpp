#include <iostream>
#include "tools.h"
#include "float.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
    VectorXd rmse(4);
    rmse << 0,0,0,0;

    // check the validity of the following inputs:
    //  * the estimation vector size should not be zero
    //  * the estimation vector size should equal ground truth vector size
    if(estimations.size() != ground_truth.size()
       || estimations.size() == 0){
        return rmse;
    }

    //accumulate squared residuals
    for(unsigned int i=0; i < estimations.size(); ++i){

        VectorXd residual = estimations[i] - ground_truth[i];

        //coefficient-wise multiplication
        residual = residual.array()*residual.array();
        rmse += residual;
    }

    //calculate the mean
    rmse = rmse/estimations.size();

    //calculate the squared root
    rmse = rmse.array().sqrt();

    //return the result
    return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    MatrixXd Hj(3,4);
    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    //check division by zero
    float px2 = px * px;
    float py2 = py * py;

    if(px2 + py2  == 0) {
        //lim x / sqrt(x*x) -> +- 1 as x -> 0
        //As we can not determine sign let's go with the mean -> 0
        Hj(0,0) = 0.0;
        Hj(0,1) = 0.0;
        Hj(0,2) = 0.0;
        Hj(0,3) = 0.0;

        //lim 1 / x as x -> 0 -> +- inf depending on from where we approach 0 -> Let's go with the mean - > 0
        Hj(1,0) = 0.0;
        Hj(1,1) = 0.0;
        Hj(1,2) = 0.0;
        Hj(1,3) = 0.0;

        //lim x^2 / (x^2)^(3/2) as x -> 0 -> + inf. Let's settle with DBL_MAX for this one

        Hj(2,0) = DBL_MAX;
        Hj(2,1) = DBL_MAX;
        Hj(2,2) = 0.0;
        Hj(2,3) = 0.0;


        return Hj;
    }
    //compute the Jacobian matrix
    Hj(0,0) = px / pow(px2+py2, 0.5);
    Hj(0,1) = py / pow(px2+py2, 0.5);
    Hj(0,2) = 0.0;
    Hj(0,3) = 0.0;

    Hj(1,0) = -py/(px2 + py2);
    Hj(1,1) = px/(px2+py2);
    Hj(1,2) = 0.0;
    Hj(1,3) = 0.0;

    Hj(2,0) = py*(vx*py - vy*px)/pow(px2+py2, 1.5);
    Hj(2,1) = px*(vy*px - vx*py)/pow(px2+py2, 1.5);
    Hj(2,2) = px/pow(px2+py2,0.5);
    Hj(2,3) = py/pow(px2+py2,0.5);

    return Hj;

}

double Tools::NormalizeFi(double fi) {
    double pi = 3.14159;
    if(fi < -pi) {
        return NormalizeFi(fi +2 * pi);
    }
    if(fi > pi) {
        return NormalizeFi(fi - 2 * pi);
    }
    return fi;
}
