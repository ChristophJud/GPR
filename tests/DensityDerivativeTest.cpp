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
#include <string>
#include <map>

#include "Prior.h"

void Test1(){
    /*
     * Test 1: derivative test of log gamma density
     */
    std::cout << "Test 1: log gamma derivative... " << std::flush;

    typedef gpr::LogGammaDensity<double>       LogGammaDensityType;
    double h=0.001;
    unsigned counter = 0;
    double err = 0;

    unsigned false_counter = 0;

    for(double alpha=0.2; alpha<10; alpha+=0.2){
        for(double beta=0.2; beta<10; beta+=0.2){
            for(double x=h/2+0.01; x<30; x+=0.1){
                if(x==1) continue;

                LogGammaDensityType lgd(alpha, beta);

                // analytical derivative
                double D = lgd.GetDerivative(x);


                // central difference
                double y1 = lgd(x+h/2);
                double y2 = lgd(x-h/2);
                double cd = (y1-y2)/h;

                // count for differences which are large
                // the nummerical differentiation is very instable
                if(std::fabs(cd-D)>0.001){
                    false_counter++;
                }
                else{
                    err += std::fabs(cd-D);
                }
                counter++;
            }
        }
    }

    // if the number of cases for which the differences have been large is smaller than 2%, its ok
    if(false_counter/static_cast<double>(counter) < 0.02){
        std::cout << "\t[passed]." << std::endl;
    }
    else{
        std::cout << "\t[failed]." << std::endl;
    }
}

int main (int argc, char *argv[]){
    std::cout << "Density derivative test: " << std::endl;
    try{
        Test1();
    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}
