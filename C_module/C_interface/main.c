#include <stdio.h>
#include "vector_mtx.h"

#include "Simulation.h"
#include "SolutionSet.h"
#include <math.h>
#include <stdlib.h>

int main() {

    printf("TEST TEST TEST\n");
    // F:/KrotovHopfieldClean/paper - figure 3
    SolutionSet* fullSet = new_SolutionSet("./C_Code_FPs");//("F:/KrotovHopfieldClean/Figure 5/C_Code_FPs");

    for (int n_ = 2; n_ < 71*10; ++n_) {
        printf("%d\n", n_);
        double n = 1.0 + n_/10.0;
        Simulation *sim_pb = new_simulation(753, 494, 719, 1.0/(4.0*1024.0), 700/(pow(2.0, 1.0/n)), n, POSITIVE_BETA, fullSet); //1.0*1024.0
        Simulation__CUDAfindSolutions(sim_pb);//Simulation__CUDAfindSolutions(sim_pb);
        free_simulation(sim_pb);

        if (n < 7) {
            Simulation *sim_nb = new_simulation(753, 494, 719, 1.0/(4.0*1024.0), 700/(pow(2.0, 1.0/n)), n, NEGATIVE_BETA, fullSet);
            Simulation__CUDAfindSolutions(sim_pb);//Simulation__CUDAfindSolutions(sim_nb);
            free_simulation(sim_nb);
        }
    }

    SolutionSet__saveSolutions(fullSet);

    return 0;
}
