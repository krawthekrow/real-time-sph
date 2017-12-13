#include "Shaders.h"

const char *Shaders::VERT_DONOTHING =
#include "DoNothing.vert"
    ;
const char *Shaders::VERT_TRANSFERDENSITY =
#include "TransferDensity.vert"
    ;
const char *Shaders::VERT_TRANSFORMPOINTS =
#include "TransformPoints.vert"
    ;
const char *Shaders::VERT_POSITIONQUAD =
#include "PositionQuad.vert"
    ;
const char *Shaders::GEOM_MAKEBILLBOARDS =
#include "MakeBillboards.geom"
    ;
const char *Shaders::FRAG_DONOTHING =
#include "DoNothing.frag"
    ;
const char *Shaders::FRAG_DRAWSPHERE =
#include "DrawSphere.frag"
    ;
const char *Shaders::FRAG_DRAWTEXTURE =
#include "DrawTexture.frag"
    ;
const char *Shaders::FRAG_DRAWTEXTUREWITHDEPTH =
#include "DrawTextureWithDepth.frag"
    ;
const char *Shaders::FRAG_RENDERFLUID =
#include "RenderFluid.frag"
    ;
const char *Shaders::FRAG_BILATERALFILTER =
#include "BilateralFilter.frag"
    ;
const char *Shaders::FRAG_BILATERALFILTERVERT =
#include "BilateralFilterVert.frag"
    ;
const char *Shaders::FRAG_BILATERALFILTERHORZ =
#include "BilateralFilterHorz.frag"
    ;
