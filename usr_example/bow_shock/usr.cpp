#include <iostream>
#include <fstream>
#include <fenv.h>

#include "../../src/device/general.h"
#include "../../src/comm/mpi/mpi.h"
#include "../../src/mesh/mesh_enroll.h"
#include "../../src/modules/chemistry/chemistry.h"
#include "../../src/utilities/mapping/loop.h"
#include "../../src/user/probgen.h"

#include "hydro/cycle.h"
#include "hydro/hydro.h"

namespace prob
{
////////////////////////////////////////////////////////////
// Interface

void run( int argc, char * argv[  ] )
{
    mesh:: mesh_t mesh;
    mesh.enroll_device    <          device_t > (  );
#ifdef    __MPI__
    mesh.enroll_comm      < comm      ::mpi_t > (  );
    mesh.enroll_binary_io < binary_io ::mpi_t > (  );
#endif // __MPI__
    mesh.enroll_cycle     < friction::cycle_t > (  );
    mesh.enroll_module    < friction::hydro_t > (  );
    
    input args;
    args.set_comm( mesh.p_com );
    cmd    ( argc, argv, args );
    if( args.get< bool >( "cycle", "except_ferr", false ) )
        feenableexcept
            ( FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW );
    
    mesh.  init   ( args );
    return mesh.evolve(  );
}

};                              // namespace prob
