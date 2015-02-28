/*
 * Copyright 2015 Christoph Jud (christoph.jud@unibas.ch)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iostream>
#include <memory>
#include <ctime>
#include <vector>

#include <boost/random.hpp>

#include "GaussianProcess.h"
#include "Kernel.h"

using namespace gpr;

typedef GaussianKernel<double>		KernelType;
typedef std::shared_ptr<KernelType> KernelTypePointer;
typedef GaussianProcess<double> GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType> GaussianProcessTypePointer;

typedef GaussianProcessType::VectorType VectorType;
typedef GaussianProcessType::MatrixType MatrixType;

VectorType GetRandomVector(unsigned n){
    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, 1);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);

    VectorType v = VectorType::Zero(n);
    for (unsigned i=0; i < n; i++) {
        v(i) = r();
    }
    return v;
}

void ExportPythonVariable(const VectorType& a, const std::string& var_name){
    std::cout << var_name << " = np.array([";
    for(unsigned i=0; i<a.rows(); i++){
        std::cout << a[i] << ", ";
    }
    std::cout << "])" << std::endl;
}

void Test1(){
    // create ground truth gp
    KernelTypePointer k(new KernelType(1));
    GaussianProcessTypePointer gp(new GaussianProcessType(k));
    gp->SetSigma(0.001);

    // add some landmarks
    gp->AddSample(VectorType::Ones(1),  VectorType::Zero(1));
    gp->AddSample(VectorType::Ones(1)*2,VectorType::Ones(1));
    gp->AddSample(VectorType::Ones(1)*3,VectorType::Ones(1)*0.5);
    gp->AddSample(VectorType::Ones(1)*4,VectorType::Ones(1));
    gp->Initialize();


    // create sample points of ground truth gp
    unsigned num_samples = 20;
    VectorType x = (VectorType::Random(num_samples).array()+1)*2.5   ; // 30 uniform samples in the interval [0,5]
    VectorType y = VectorType::Zero(num_samples);
    VectorType y_noisefree = VectorType::Zero(num_samples);
    for(unsigned i=0; i<num_samples; i++){
        y[i] = gp->Predict(VectorType::Ones(1)*x[i])[0] + GetRandomVector(1)[0]*0.1;
        y_noisefree[i] = gp->Predict(VectorType::Ones(1)*x[i])[0];
    }

    VectorType m = VectorType::Zero(70);
    VectorType x2 = (VectorType::Random(m.rows()).array()+1)*2.5;
    for(unsigned i=1; i<=1000; i++){
        // sample a sigma
        static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
        static boost::normal_distribution<> dist(1, 1);
        static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);
        double sigma = std::fabs(r());

        KernelTypePointer k(new KernelType(sigma));
        GaussianProcessTypePointer gp(new GaussianProcessType(k));
        gp->SetSigma(0.001);
        for(unsigned j=0; j<x.rows(); j++){
            VectorType sx = VectorType::Zero(1);
            VectorType sy = VectorType::Zero(1);
            sx[0] = x[j];
            sy[0] = y[j];
            gp->AddSample(sx, sy);
        }
        gp->Initialize();

        for(unsigned j=0; j<m.rows(); j++){
            m[j] += gp->Predict(VectorType::Ones(1)*x2[j])[0];
        }
    }
    for(unsigned j=0; j<m.rows(); j++){
        m[j] /= 1000;
    }


    std::cout << "import numpy as np" << std::endl;
    std::cout << "import pylab as plt" << std::endl;

    ExportPythonVariable(x, "x");
    ExportPythonVariable(y, "y");
    ExportPythonVariable(y_noisefree, "ynf");


    ExportPythonVariable(x2, "x2");
    ExportPythonVariable(m, "m");

    std::cout << "plt.plot(x,y, '.b')" << std::endl;
    std::cout << "plt.plot(x,ynf, '.r')" << std::endl;
    std::cout << "plt.plot(x2,m, '.k')" << std::endl;
    std::cout << "plt.show()" << std::endl;

}



int main (int argc, char *argv[]){
    //std::cout << "Gaussian process parameter estimation test: " << std::endl;
    Test1();

    return 0;
}

