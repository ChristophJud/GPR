
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
#include <cmath>

#include <boost/random.hpp>

#include "GaussianProcess.h"
#include "Kernel.h"

using namespace gpr;

typedef double ScalarType;
typedef Kernel<ScalarType>  KernelType;
typedef std::shared_ptr<KernelType> KernelTypePointer;
typedef GaussianProcess<ScalarType> GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType> GaussianProcessTypePointer;

typedef typename GaussianProcessType::VectorType VectorType;
typedef typename GaussianProcessType::MatrixType MatrixType;

void Test1(){
    /*
     * Test 1: derivative test of kernels
     */
    std::cout << "Test 1.1: gaussian kernel derivative... " << std::flush;

    VectorType x = VectorType::Zero(2);
    x(0) = 0.1; x(1) = 0.5;

    VectorType y = VectorType::Zero(2);
    y(0) = -0.1; y(1) = 0.8;


    // typedefs
    typedef GaussianKernel<ScalarType>    GaussianKernelType;
    typedef std::shared_ptr<GaussianKernelType>                 GaussianKernelTypePointer;

    double h = 0.001;
    VectorType err = VectorType::Zero(2);
    unsigned counter = 0;
    for(double sigma=0.1; sigma<10; sigma+=0.4){
        for(double scale=0.1; scale<3; scale+=0.8){
            // analytical derivative
            GaussianKernelTypePointer gk(new GaussianKernelType(sigma, scale));
            VectorType D = gk->GetDerivative(x,y);

            // scale central difference
            GaussianKernelTypePointer gk1_scale(new GaussianKernelType(sigma, scale+h/2));
            GaussianKernelTypePointer gk2_scale(new GaussianKernelType(sigma, scale-h/2));
            double cd_scale = (*gk1_scale)(x,y) - (*gk2_scale)(x,y);
            cd_scale/=h;

            // sigma central difference
            GaussianKernelTypePointer gk1_sigma(new GaussianKernelType(sigma+h/2, scale));
            GaussianKernelTypePointer gk2_sigma(new GaussianKernelType(sigma-h/2, scale));
            double cd_sigma = (*gk1_sigma)(x,y) - (*gk2_sigma)(x,y);
            cd_sigma/=h;

            err[0] += std::fabs(cd_sigma-D[0]);
            err[1] += std::fabs(cd_scale-D[1]);

            counter++;
        }
    }


    if(err[0]/counter < 1e-5 && err[1]/counter < 1e-12){
        std::cout << "\t[passed]." << std::endl;
    }
    else{
        std::cout << "\t[failed] with an error (sigma) " << err[0]/counter << " and (scale) " << err[1]/counter << std::endl;
    }
    return;


}

void Test1_2(){
    /*
     * Test 1.2: derivative test of kernels
     */
    std::cout << "Test 1.2: gaussian kernel derivative with exponentiated parameters... " << std::flush;

    VectorType x = VectorType::Zero(2);
    x(0) = 0.1; x(1) = 0.5;

    VectorType y = VectorType::Zero(2);
    y(0) = -0.1; y(1) = 0.8;


    // typedefs
    typedef GaussianExpKernel<ScalarType>           GaussianExpKernelType;
    typedef std::shared_ptr<GaussianExpKernelType>  GaussianExpKernelTypePointer;

    double h = 0.001;
    VectorType err = VectorType::Zero(2);
    unsigned counter = 0;
    for(double sigma=0.1; sigma<10; sigma+=0.4){
        for(double scale=0.1; scale<3; scale+=0.8){
            // analytical derivative
            GaussianExpKernelTypePointer gk(new GaussianExpKernelType(sigma, scale));
            VectorType D = gk->GetDerivative(x,y);

            // scale central difference
            GaussianExpKernelTypePointer gk1_scale(new GaussianExpKernelType(sigma, scale+h/2));
            GaussianExpKernelTypePointer gk2_scale(new GaussianExpKernelType(sigma, scale-h/2));
            double cd_scale = (*gk1_scale)(x,y) - (*gk2_scale)(x,y);
            cd_scale/=h;

            // sigma central difference
            GaussianExpKernelTypePointer gk1_sigma(new GaussianExpKernelType(sigma+h/2, scale));
            GaussianExpKernelTypePointer gk2_sigma(new GaussianExpKernelType(sigma-h/2, scale));
            double cd_sigma = (*gk1_sigma)(x,y) - (*gk2_sigma)(x,y);
            cd_sigma/=h;

            err[0] += std::fabs(cd_sigma-D[0]);
            err[1] += std::fabs(cd_scale-D[1]);

            counter++;
        }
    }

    if(err[0]/counter < 1e-6 && err[1]/counter < 1e-3){
        std::cout << "\t[passed]." << std::endl;
    }
    else{
        std::cout << "\t[failed] with an error (sigma) " << err[0]/counter << " and (scale) " << err[1]/counter << std::endl;
    }
    return;
}

void Test2(){
    /*
     * Test 2: derivative test of kernels
     */
    std::cout << "Test 2: white kernel derivative... " << std::flush;

    VectorType x = VectorType::Zero(2);
    x(0) = 0.1; x(1) = 0.5;

    VectorType y = VectorType::Zero(2);
    y(0) = -0.1; y(1) = 0.8;


    // typedefs
    typedef WhiteKernel<ScalarType>              KernelType;
    typedef std::shared_ptr<KernelType>          KernelTypePointer;

    double h = 0.1;
    VectorType err1 = VectorType::Zero(1);
    VectorType err2 = VectorType::Zero(1);
    unsigned counter = 0;

    for(double scale=0.1; scale<3; scale+=0.8){
        // analytical derivative
        KernelTypePointer k(new KernelType(scale));
        VectorType D1 = k->GetDerivative(x,y);
        VectorType D2 = k->GetDerivative(x,x);

        // scale central difference
        KernelTypePointer k1_scale(new KernelType(scale+h/2));
        KernelTypePointer k2_scale(new KernelType(scale-h/2));
        double cd_scale1 = (*k1_scale)(x,y) - (*k2_scale)(x,y);
        cd_scale1/=h;
        double cd_scale2 = (*k1_scale)(x,x) - (*k2_scale)(x,x);
        cd_scale2/=h;

        err1[0] += std::fabs(cd_scale1-D1[0]);
        err2[0] += std::fabs(cd_scale2-D2[0]);
        counter++;
    }

    if(err1[0]/counter == 0 && err2[0]/counter < 1e-13){
        std::cout << "\t[passed]." << std::endl;
    }
    else{
        std::cout << "\t[failed]." << std::endl;
    }
}

void Test3(){
    /*
     * Test 3: derivative test of kernels
     */
    std::cout << "Test 3: rational quadratic kernel derivative... " << std::flush;

    VectorType x = VectorType::Zero(2);
    x(0) = 0.1; x(1) = 0.5;

    VectorType y = VectorType::Zero(2);
    y(0) = -0.1; y(1) = 0.8;


    // typedefs
    typedef RationalQuadraticKernel<ScalarType>              KernelType;
    typedef std::shared_ptr<KernelType>          KernelTypePointer;

    double h = 0.01;
    VectorType err = VectorType::Zero(3);
    unsigned counter = 0;

    for(double scale=0.1; scale<3; scale+=0.8){
        for(double sigma=0.2; sigma<10; sigma+=0.6){
            for(double alpha=0.1; alpha<6; alpha+=0.6){
                // analytical derivative
                KernelTypePointer k(new KernelType(scale, sigma, alpha));
                VectorType D = k->GetDerivative(x,y);

                // scale central difference
                KernelTypePointer k1_scale(new KernelType(scale+h/2, sigma, alpha));
                KernelTypePointer k2_scale(new KernelType(scale-h/2, sigma, alpha));
                double cd_scale = (*k1_scale)(x,y) - (*k2_scale)(x,y);
                cd_scale/=h;

                // sigma central difference
                KernelTypePointer k1_sigma(new KernelType(scale, sigma+h/2, alpha));
                KernelTypePointer k2_sigma(new KernelType(scale, sigma-h/2, alpha));
                double cd_sigma = (*k1_sigma)(x,y) - (*k2_sigma)(x,y);
                cd_sigma/=h;

                // alpha central difference
                KernelTypePointer k1_alpha(new KernelType(scale, sigma, alpha+h/2));
                KernelTypePointer k2_alpha(new KernelType(scale, sigma, alpha-h/2));
                double cd_alpha = (*k1_alpha)(x,y) - (*k2_alpha)(x,y);
                cd_alpha/=h;

                err[0] += std::fabs(cd_scale-D[0]);
                err[1] += std::fabs(cd_sigma-D[1]);
                err[2] += std::fabs(cd_alpha-D[2]);
                counter++;
            }
        }
    }

    // sigma and alpha are not that stable with central difference

    if(err[0]/counter < 1e-13 && err[1]/counter < 0.001 && err[2]/counter < 1e-4){
        std::cout << "\t\t[passed]." << std::endl;
    }
    else{
        std::cout << "\t\t[failed]." << std::endl;
    }
}

void Test4(){
    /*
     * Test 4: derivative test of kernels
     */
    std::cout << "Test 4: periodic kernel derivative... " << std::flush;

    VectorType x = VectorType::Zero(2);
    x(0) = 0.1; x(1) = 0.5;

    VectorType y = VectorType::Zero(2);
    y(0) = -0.1; y(1) = 0.8;


    // typedefs
    typedef PeriodicKernel<ScalarType>              KernelType;
    typedef std::shared_ptr<KernelType>          KernelTypePointer;

    double h = 0.01;
    VectorType err = VectorType::Zero(3);
    unsigned counter = 0;

    for(double scale=0.1; scale<3; scale+=0.8){
        for(double b=0.1; b<5*M_PI; b+=0.3){
            for(double sigma=0.1; sigma<4; sigma+=0.1){
                // analytical derivative
                KernelTypePointer k(new KernelType(scale, b, sigma));
                VectorType D = k->GetDerivative(x,y);

                // scale central difference
                KernelTypePointer k1_scale(new KernelType(scale+h/2, b, sigma));
                KernelTypePointer k2_scale(new KernelType(scale-h/2, b, sigma));
                double cd_scale = (*k1_scale)(x,y) - (*k2_scale)(x,y);
                cd_scale/=h;

                // sigma central difference
                KernelTypePointer k1_b(new KernelType(scale, b+h/2, sigma));
                KernelTypePointer k2_b(new KernelType(scale, b-h/2, sigma));
                double cd_b = (*k1_b)(x,y) - (*k2_b)(x,y);
                cd_b/=h;

                // alpha central difference
                KernelTypePointer k1_sigma(new KernelType(scale, b, sigma+h/2));
                KernelTypePointer k2_sigma(new KernelType(scale, b, sigma-h/2));
                double cd_sigma = (*k1_sigma)(x,y) - (*k2_sigma)(x,y);
                cd_sigma/=h;

                err[0] += std::fabs(cd_scale-D[0]);
                err[1] += std::fabs(cd_b-D[1]);
                err[2] += std::fabs(cd_sigma-D[2]);
                counter++;
            }
        }
    }

    if(err[0]/counter < 1e-13 && err[1]/counter < 1e-5 && err[2]/counter < 0.001){
        std::cout << "\t[passed]." << std::endl;
    }
    else{
        std::cout << "\t[failed]." << std::endl;
    }
}

void Test5(){
    /*
     * Test 5: derivative test of kernels
     */
    std::cout << "Test 5: sum of gaussian and periodic kernel derivative... " << std::flush;

    VectorType x = VectorType::Zero(2);
    x(0) = 0.1; x(1) = 0.5;

    VectorType y = VectorType::Zero(2);
    y(0) = -0.1; y(1) = 0.8;


    // typedefs
    typedef SumKernel<ScalarType>                ProductKernelType;
    typedef std::shared_ptr<ProductKernelType>       ProductKernelTypePointer;
    typedef GaussianKernel<ScalarType>           GaussianKernelType;
    typedef std::shared_ptr<GaussianKernelType>  GaussianKernelTypePointer;
    typedef PeriodicKernel<ScalarType>           PeriodicKernelType;
    typedef std::shared_ptr<PeriodicKernelType>  PeriodicKernelTypePointer;

    double h = 0.01;
    VectorType err = VectorType::Zero(5);
    unsigned counter = 0;

    for(double gscale=0.1; gscale<5; gscale+=0.8){
        for(double gsigma=0.1; gsigma<6; gsigma+=0.4){
            for(double pscale=0.1; pscale<4; pscale+=0.8){
                for(double b=0.1; b<5*M_PI; b+=0.4){
                    for(double psigma=0.2; psigma<6; psigma+=0.3){

                        // analytical derivative
                        GaussianKernelTypePointer gk(new GaussianKernelType(gsigma, gscale));
                        PeriodicKernelTypePointer pk(new PeriodicKernelType(pscale, b, psigma));
                        ProductKernelTypePointer k(new ProductKernelType(gk,pk));
                        VectorType D = k->GetDerivative(x,y);

                        // gaussian scale central difference
                        GaussianKernelTypePointer gk1_gscale(new GaussianKernelType(gsigma, gscale+h/2));
                        GaussianKernelTypePointer gk2_gscale(new GaussianKernelType(gsigma, gscale-h/2));
                        ProductKernelTypePointer k1_gscale(new ProductKernelType(gk1_gscale, pk));
                        ProductKernelTypePointer k2_gscale(new ProductKernelType(gk2_gscale, pk));
                        double cd_gscale = (*k1_gscale)(x,y) - (*k2_gscale)(x,y);
                        cd_gscale/=h;

                        // gaussian scale central difference
                        GaussianKernelTypePointer gk1_gsigma(new GaussianKernelType(gsigma+h/2, gscale));
                        GaussianKernelTypePointer gk2_gsigma(new GaussianKernelType(gsigma-h/2, gscale));
                        ProductKernelTypePointer k1_gsigma(new ProductKernelType(gk1_gsigma, pk));
                        ProductKernelTypePointer k2_gsigma(new ProductKernelType(gk2_gsigma, pk));
                        double cd_gsigma = (*k1_gsigma)(x,y) - (*k2_gsigma)(x,y);
                        cd_gsigma/=h;

                        // periodic scale central difference
                        PeriodicKernelTypePointer pk1_pscale(new PeriodicKernelType(pscale+h/2, b, psigma));
                        PeriodicKernelTypePointer pk2_pscale(new PeriodicKernelType(pscale-h/2, b, psigma));
                        ProductKernelTypePointer k1_pscale(new ProductKernelType(gk, pk1_pscale));
                        ProductKernelTypePointer k2_pscale(new ProductKernelType(gk, pk2_pscale));
                        double cd_pscale = (*k1_pscale)(x,y) - (*k2_pscale)(x,y);
                        cd_pscale/=h;

                        // periodic period length central difference
                        PeriodicKernelTypePointer pk1_pb(new PeriodicKernelType(pscale, b+h/2, psigma));
                        PeriodicKernelTypePointer pk2_pb(new PeriodicKernelType(pscale, b-h/2, psigma));
                        ProductKernelTypePointer k1_pb(new ProductKernelType(gk, pk1_pb));
                        ProductKernelTypePointer k2_pb(new ProductKernelType(gk, pk2_pb));
                        double cd_pb = (*k1_pb)(x,y) - (*k2_pb)(x,y);
                        cd_pb/=h;

                        // periodic sigma central difference
                        PeriodicKernelTypePointer pk1_psigma(new PeriodicKernelType(pscale, b, psigma+h/2));
                        PeriodicKernelTypePointer pk2_psigma(new PeriodicKernelType(pscale, b, psigma-h/2));
                        ProductKernelTypePointer k1_psigma(new ProductKernelType(gk, pk1_psigma));
                        ProductKernelTypePointer k2_psigma(new ProductKernelType(gk, pk2_psigma));
                        double cd_psigma = (*k1_psigma)(x,y) - (*k2_psigma)(x,y);
                        cd_psigma/=h;



                        err[0] += std::fabs(cd_gsigma-D[0]);
                        err[1] += std::fabs(cd_gscale-D[1]);
                        err[2] += std::fabs(cd_pscale-D[2]);
                        err[3] += std::fabs(cd_pb-D[3]);
                        err[4] += std::fabs(cd_psigma-D[4]);
                        counter++;
                    }
                }
            }
        }
    }

    if(err[0]/counter < 0.005 &&
            err[1]/counter < 1e-11 &&
            err[2]/counter < 1e-11 &&
            err[3]/counter < 1e-6 &&
            err[4]/counter < 1e-4){
        std::cout << "\t[passed]." << std::endl;
    }
    else{
        std::cout << "\t[failed]." << std::endl;
    }
}

void Test6(){
    /*
     * Test 6: derivative test of kernels
     */
    std::cout << "Test 6: product of gaussian and periodic kernel derivative... " << std::flush;

    VectorType x = VectorType::Zero(2);
    x(0) = 0.1; x(1) = 0.5;

    VectorType y = VectorType::Zero(2);
    y(0) = -0.1; y(1) = 0.8;


    // typedefs
    typedef ProductKernel<ScalarType>            ProductKernelType;
    typedef std::shared_ptr<ProductKernelType>   ProductKernelTypePointer;
    typedef GaussianKernel<ScalarType>           GaussianKernelType;
    typedef std::shared_ptr<GaussianKernelType>  GaussianKernelTypePointer;
    typedef PeriodicKernel<ScalarType>           PeriodicKernelType;
    typedef std::shared_ptr<PeriodicKernelType>  PeriodicKernelTypePointer;

    double h = 0.01;
    VectorType err = VectorType::Zero(5);
    unsigned counter = 0;

    for(double gscale=0.4; gscale<4; gscale+=0.8){
        for(double gsigma=0.1; gsigma<5; gsigma+=0.4){
            for(double pscale=0.1; pscale<4; pscale+=0.8){
                for(double b=0.1; b<4*M_PI; b+=0.4){
                    for(double psigma=0.4; psigma<5; psigma+=0.3){

                        // analytical derivative
                        GaussianKernelTypePointer gk(new GaussianKernelType(gsigma, gscale));
                        PeriodicKernelTypePointer pk(new PeriodicKernelType(pscale, b, psigma));
                        ProductKernelTypePointer k(new ProductKernelType(gk,pk));
                        VectorType D = k->GetDerivative(x,y);

                        // gaussian scale central difference
                        GaussianKernelTypePointer gk1_gscale(new GaussianKernelType(gsigma, gscale+h/2));
                        GaussianKernelTypePointer gk2_gscale(new GaussianKernelType(gsigma, gscale-h/2));
                        ProductKernelTypePointer k1_gscale(new ProductKernelType(gk1_gscale, pk));
                        ProductKernelTypePointer k2_gscale(new ProductKernelType(gk2_gscale, pk));
                        double cd_gscale = (*k1_gscale)(x,y) - (*k2_gscale)(x,y);
                        cd_gscale/=h;

                        // gaussian scale central difference
                        GaussianKernelTypePointer gk1_gsigma(new GaussianKernelType(gsigma+h/2, gscale));
                        GaussianKernelTypePointer gk2_gsigma(new GaussianKernelType(gsigma-h/2, gscale));
                        ProductKernelTypePointer k1_gsigma(new ProductKernelType(gk1_gsigma, pk));
                        ProductKernelTypePointer k2_gsigma(new ProductKernelType(gk2_gsigma, pk));
                        double cd_gsigma = (*k1_gsigma)(x,y) - (*k2_gsigma)(x,y);
                        cd_gsigma/=h;

                        // periodic scale central difference
                        PeriodicKernelTypePointer pk1_pscale(new PeriodicKernelType(pscale+h/2, b, psigma));
                        PeriodicKernelTypePointer pk2_pscale(new PeriodicKernelType(pscale-h/2, b, psigma));
                        ProductKernelTypePointer k1_pscale(new ProductKernelType(gk, pk1_pscale));
                        ProductKernelTypePointer k2_pscale(new ProductKernelType(gk, pk2_pscale));
                        double cd_pscale = (*k1_pscale)(x,y) - (*k2_pscale)(x,y);
                        cd_pscale/=h;

                        // periodic period length central difference
                        PeriodicKernelTypePointer pk1_pb(new PeriodicKernelType(pscale, b+h/2, psigma));
                        PeriodicKernelTypePointer pk2_pb(new PeriodicKernelType(pscale, b-h/2, psigma));
                        ProductKernelTypePointer k1_pb(new ProductKernelType(gk, pk1_pb));
                        ProductKernelTypePointer k2_pb(new ProductKernelType(gk, pk2_pb));
                        double cd_pb = (*k1_pb)(x,y) - (*k2_pb)(x,y);
                        cd_pb/=h;

                        // periodic sigma central difference
                        PeriodicKernelTypePointer pk1_psigma(new PeriodicKernelType(pscale, b, psigma+h/2));
                        PeriodicKernelTypePointer pk2_psigma(new PeriodicKernelType(pscale, b, psigma-h/2));
                        ProductKernelTypePointer k1_psigma(new ProductKernelType(gk, pk1_psigma));
                        ProductKernelTypePointer k2_psigma(new ProductKernelType(gk, pk2_psigma));
                        double cd_psigma = (*k1_psigma)(x,y) - (*k2_psigma)(x,y);
                        cd_psigma/=h;



                        err[0] += std::fabs(cd_gsigma-D[0]);
                        err[1] += std::fabs(cd_gscale-D[1]);
                        err[2] += std::fabs(cd_pscale-D[2]);
                        err[3] += std::fabs(cd_pb-D[3]);
                        err[4] += std::fabs(cd_psigma-D[4]);
                        counter++;
                    }
                }
            }
        }
    }

    if(err[0]/counter < 0.009 &&
            err[1]/counter < 1e-11 &&
            err[2]/counter < 1e-11 &&
            err[3]/counter < 1e-5 &&
            err[4]/counter < 0.001){
        std::cout << "\t[passed]." << std::endl;
    }
    else{
        std::cout << "\t[failed]." << std::endl;
    }
}

int main (int argc, char *argv[]){
    std::cout << "Product kernel test: " << std::endl;
    try{
        Test1();
        Test1_2();
        Test2();
        Test3();
        Test4();
        Test5();
        Test6();
    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}


