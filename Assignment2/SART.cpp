/*
    Defines a callable SART Method Function for MATLAB

    To run, simply compile the binary by running 'mex SART.cpp', then
    call the function via 'SART(args...)'
*/
#include <iostream>
#include "mex.hpp"
#include "mexAdapter.hpp"

// Define a Mex Function for calling SART method
class MexFunction : public matlab::mex::Function {
    public:
        // Main Entrypoint
        void operator()(matlab::mex::ArgumentList outputs, matlab::mex::ArgumentList inputs) {
            // Store inputs
            const matlab::data::SparseArray<double> Kt = inputs[0]; // Transposed K
            const matlab::data::TypedArray<double> Knorms = inputs[1]; // K norms
            const matlab::data::TypedArray<double> d = inputs[2]; // Data Vector
            const matlab::data::TypedArray<double> p = inputs[3]; // projection labels

            // Create Array Factory
            matlab::data::ArrayFactory factory;

            // Create a TypedArray to store m for output, residuals, and error
            matlab::data::TypedArray<double> m = factory.createArray<double>({4096,1000});

            /*
                Create C++ variables for miscellaneous tasks (Faster than using MATLAB types)
            */

            // 3D array to store rows of K, set to max row size (~126), store, element, index, endflag respectively
            double*** Krow = Create3DArray(7328,130,3);
            double* m_recon = new double[4096]; // array to hold the iterative m reconstruction, we copy to MATLAB array later
            double* m_temp = new double[4096]; // temp m for each projection angle
            double* di = new double[7328]; // store element of data array
            double* Kinorm = new double[7328]; // store elements of the K row norm^2
            int* pi = new int[7328]; // store projection labels
            double ip; // for storing inner product calculations
            int t; // multiple index for projection angle
            int q; // keeps track of number of projections to average over

            // Store the rows of the K matrix in more efficient form
            std::cout << "Copying data over to C data types..." << std::endl;
            GetRow(Krow,&Kt);

            // Convert to C data types
            for (int i=0; i<7328; i++) {
                di[i] = d[i];
                Kinorm[i] = Knorms[i];
                pi[i] = int(p[i]);
            }

            // Initialize m arrays to 0
            for (int i=0; i<4096; i++) {
                m_temp[i] = 0;
                m_recon[i] = 0;
            }

            // Do SART method
            for (int j=0; j<1000; j++) {
                std::cout << "Iteration #: " << j << std::endl;

                // reset the angle index
                t = 1;
                q = 0;

                // loop over each row (I do to 7329 so the last projection can be accounted for)
                for (int i=0; i<7329; i++) {
                    // quit if we passed the last iterate
                    if (i == 7328)
                        break;
    
                    // Get current row;
                    double** row = Krow[i];

                    // calculate inner product between m and d
                    InnerProduct(ip,row,m_recon);

                    /*
                        This is different from ART b/c we are only updating m_recon
                        after each projection angle. This means we need to keep track
                        of what angle we are currently on, and add a conditional for
                        when to update m_recon
                    */

                    // check the current row projection label (90 b/c every new angle = 90)
                    if ( pi[i] > t*90) { // condition for new angle
                        // update m since we finish with current projection angle
                        for (int l=1; l<4096; l++) {
                                m_recon[l] += m_temp[l]/q;
                                m_temp[l] = 0; // reset m_temp for next angle
                        }
                        t++; // move to next angle
                        q = 0; // reset average
                    }

                    // calculate m temp (where row is not 0)
                    for (int k=0; k<130; k++) {
                        if (row[k][2] == 1) // break if endflag reached
                            break;
                        m_temp[int(row[k][1])] += ((di[i] - ip)/Kinorm[i])*row[k][0];
                        q++; // increment average
                    }
                }

                // Save m_recon into matlab array
                for (int k=0; k<4096; k++) {
                    m[k][j] = m_recon[k];
                }
            }

            // Clean-up
            for (int i=0; i<7328; i++) {
                for (int j=0; j<130; j++) {
                        delete [] Krow[i][j];
                }
                delete [] Krow[i];
            }
            delete [] Krow;
            delete [] m_recon;
            delete [] Kinorm;
            delete [] pi;

            // Retrun outputs
            outputs[0] = m;
        }

        // Define a function that obtains a row from a sparse matrix given an index
        void GetRow(double*** Krow, const matlab::data::SparseArray<double>* Kt) {
            /*
                The idea here is that we have a sparse matrix where we can iterate over each non-zero element. We input the
                transposed matrix so we can simply iterate over the array ordering without any issues. If you examine the number
                of elements in each row of K, you can see that it never exceeds ~126. So I can represent the data in a much smaller
                array, rather than traversing 4096 elements.

                Essentially the output is a lookup table for the row, where I only save out the index of the row and it's value.

                Since each row of K is dynamic in size, I need a way to track the last value of the row. This lets me know
                when to stop iterating over the row. I call this the 'endflag' that is set to 1, that gives me a signal for the
                last element of the row.
            */
            // Create an iterator to traverse through K
            matlab::data::TypedIterator<const double> it = Kt->begin();

            // Loop through each row
            for (int k=0; k<7328; k++) {
                // set initial index for storing row data
                int i = 0;

                // Use iterator of SparseArray to get relevent row indices
                for (; it != Kt->end(); ++it) {
                    if (Kt->getIndex(it).second > k) {// done with row, set endflag and break
                        Krow[k][i][2] = 1; // endflag
                        break;
                    }
                    if (Kt->getIndex(it).second == k) { // Assign the value to the row if it matches the row we are looking for
                        Krow[k][i][0] = it[0]; // element value
                        Krow[k][i][1] = Kt->getIndex(it).first; // element index
                        Krow[k][i][2] = 0; // endflag
                        i++; // increment to the next row for storage
                    }
                }
            }
        }

        // Define a function for doing inner product
        void InnerProduct(double& ip, double** row, double* m) {
            // Reset inner product
            ip = 0;

            // Do inner product betwen row of K and m
            for (int i=0; i<130; i++ ) {
                if (row[i][2] == 1) // break if endflag reached
                    break;
                ip += row[i][0] * m[int(row[i][1])];
            }
        }

        // Define a function produce a 3D array
        double*** Create3DArray(int x, int y, int z) {
            double*** array = new double**[x];
            for (int i=0; i<x; i++) {
                array[i] = new double*[y];
                for (int j=0; j<y; j++) {
                    array[i][j] = new double[z];
                    for (int k=0; k<z; k++) {
                        array[i][j][k] = 0;
                    }
                }
            }
            return array;
        }
};
