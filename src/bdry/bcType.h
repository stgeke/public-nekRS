#if !defined(nekrs_bctype_h_)
#define nekrs_bctype_h_

#if 0
header file used in C++ and Fortran files

index-1 contiguous IDs
lower value wins on shared edges/corners
#endif

#define p_bcTypeW 1
#define p_bcTypeV 2

#define p_bcTypeSYMX 3
#define p_bcTypeSYMY 4
#define p_bcTypeSYMZ 5
#define p_bcTypeSYM  6

#define p_bcTypeSHLX 8
#define p_bcTypeSHLY 9
#define p_bcTypeSHLZ 10
#define p_bcTypeSHL  11

#define p_bcTypeONX 12
#define p_bcTypeONY 13
#define p_bcTypeONZ 14

#define p_bcTypeON 15
#define p_bcTypeO 16

#define p_bcTypeMV 17 

#define p_velNBcType 18

#define p_bcTypeS 1
#define p_bcTypeF0 2
#define p_bcTypeF 3

#define p_scalNBcType 3

#endif
