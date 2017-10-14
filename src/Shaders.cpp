#include "Shaders.h"

const char *Shaders::VERT_SIMPLE =
#include "Simple.vert"
    ;
const char *Shaders::VERT_GRIDDER =
#include "Gridder.vert"
    ;
const char *Shaders::FRAG_SIMPLE =
#include "Simple.frag"
    ;
const char *Shaders::GEOM_VOXELMESHER =
#include "VoxelMesher.geom"
    ;
const char *Shaders::COMP_NOISE3D =
#include "Noise3D.comp"
    ;
const char *Shaders::VERT_DONOTHING =
#include "DoNothing.vert"
    ;
const char *Shaders::FRAG_DONOTHING =
#include "DoNothing.frag"
    ;
const char *Shaders::GEOM_MAKEBILLBOARDS =
#include "MakeBillboards.geom"
    ;
const char *Shaders::FRAG_DRAWSPHERE =
#include "DrawSphere.frag"
    ;
