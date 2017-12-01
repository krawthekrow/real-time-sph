#pragma once

class Shaders {
public:
    static const char *VERT_DONOTHING;
    static const char *VERT_TRANSFERDENSITY;
    static const char *VERT_SIMPLE;
    static const char *VERT_TRANSFORMPOINTS;
    static const char *VERT_GRIDDER;
    static const char *VERT_POSITIONQUAD;
    static const char *GEOM_VOXELMESHER;
    static const char *GEOM_MAKEBILLBOARDS;
    static const char *GEOM_MAKEBILLBOARDSZPREPASS;
    static const char *FRAG_SIMPLE;
    static const char *FRAG_DONOTHING;
    static const char *FRAG_DRAWDISCZPREPASS;
    static const char *FRAG_DRAWSPHEREZPREPASS;
    static const char *FRAG_DRAWSPHERE;
    static const char *FRAG_DRAWTEXTURE;
    static const char *FRAG_DRAWTEXTUREWITHDEPTH;
    static const char *COMP_NOISE3D;
};
