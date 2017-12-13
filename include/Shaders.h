#pragma once

class Shaders {
public:
    static const char *VERT_DONOTHING;
    static const char *VERT_TRANSFERDENSITY;
    static const char *VERT_TRANSFORMPOINTS;
    static const char *VERT_POSITIONQUAD;
    static const char *GEOM_MAKEBILLBOARDS;
    static const char *FRAG_DONOTHING;
    static const char *FRAG_DRAWSPHERE;
    static const char *FRAG_DRAWTEXTURE;
    static const char *FRAG_DRAWTEXTUREWITHDEPTH;
    static const char *FRAG_RENDERFLUID;
    static const char *FRAG_BILATERALFILTER;
    static const char *FRAG_BILATERALFILTERVERT;
    static const char *FRAG_BILATERALFILTERHORZ;
};
