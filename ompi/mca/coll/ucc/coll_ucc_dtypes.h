/*
 * Copyright (c) 2021 Mellanox Technologies. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */
#ifndef COLL_UCC_DTYPES_H
#define COLL_UCC_DTYPES_H
#include "ompi/datatype/ompi_datatype.h"
#include "ompi/datatype/ompi_datatype_internal.h"
#include "ompi/mca/op/op.h"
#include <ucc/api/ucc.h>
#include "coll_ucc.h"

#define COLL_UCC_DT_UNSUPPORTED ((ucc_datatype_t)-1)
#define COLL_UCC_OP_UNSUPPORTED ((ucc_reduction_op_t)-1)

static ucc_datatype_t ompi_datatype_2_ucc_dt[OPAL_DATATYPE_MAX_PREDEFINED] = {
    [OPAL_DATATYPE_LOOP]                      = COLL_UCC_DT_UNSUPPORTED,
    [OPAL_DATATYPE_END_LOOP]                  = COLL_UCC_DT_UNSUPPORTED,
    [OPAL_DATATYPE_LB]                        = COLL_UCC_DT_UNSUPPORTED,
    [OPAL_DATATYPE_UB]                        = COLL_UCC_DT_UNSUPPORTED,
    [OPAL_DATATYPE_INT1]                      = UCC_DT_INT8,
    [OPAL_DATATYPE_INT2]                      = UCC_DT_INT16,
    [OPAL_DATATYPE_INT4]                      = UCC_DT_INT32,
    [OPAL_DATATYPE_INT8]                      = UCC_DT_INT64,
    [OPAL_DATATYPE_INT16]                     = UCC_DT_INT128,
    [OPAL_DATATYPE_UINT1]                     = UCC_DT_UINT8,
    [OPAL_DATATYPE_UINT2]                     = UCC_DT_UINT16,
    [OPAL_DATATYPE_UINT4]                     = UCC_DT_UINT32,
    [OPAL_DATATYPE_UINT8]                     = UCC_DT_UINT64,
    [OPAL_DATATYPE_UINT16]                    = UCC_DT_UINT128,
    [OPAL_DATATYPE_FLOAT2]                    = UCC_DT_FLOAT16,
    [OPAL_DATATYPE_FLOAT4]                    = UCC_DT_FLOAT32,
    [OPAL_DATATYPE_FLOAT8]                    = UCC_DT_FLOAT64,
    [OPAL_DATATYPE_FLOAT12]                   = COLL_UCC_DT_UNSUPPORTED,
    [OPAL_DATATYPE_BOOL]                      = COLL_UCC_DT_UNSUPPORTED,
    [OPAL_DATATYPE_WCHAR]                     = COLL_UCC_DT_UNSUPPORTED,
    [OPAL_DATATYPE_SHORT_FLOAT_COMPLEX]       = COLL_UCC_DT_UNSUPPORTED,
#if SIZEOF_LONG == 4
    [OPAL_DATATYPE_LONG]                      = UCC_DT_INT32,
    [OPAL_DATATYPE_UNSIGNED_LONG]             = UCC_DT_UINT32,
#elif SIZEOF_LONG == 8
    [OPAL_DATATYPE_LONG]                      = UCC_DT_INT64,
    [OPAL_DATATYPE_UNSIGNED_LONG]             = UCC_DT_UINT64,
#endif
#if UCC_HAVE_COMPLEX_AND_FLOAT128_DT
    [OPAL_DATATYPE_FLOAT16]                   = UCC_DT_FLOAT128,
    #if SIZEOF_FLOAT__COMPLEX == 8
        [OPAL_DATATYPE_FLOAT_COMPLEX]         = UCC_DT_FLOAT32_COMPLEX,
    #else
        [OPAL_DATATYPE_FLOAT_COMPLEX]         = COLL_UCC_DT_UNSUPPORTED,
    #endif
    #if SIZEOF_DOUBLE__COMPLEX == 16
        [OPAL_DATATYPE_DOUBLE_COMPLEX]        = UCC_DT_FLOAT64_COMPLEX,
    #else
        [OPAL_DATATYPE_DOUBLE_COMPLEX]        = COLL_UCC_DT_UNSUPPORTED,
    #endif
    #if SIZEOF_LONG_DOUBLE__COMPLEX == 32
        [OPAL_DATATYPE_LONG_DOUBLE_COMPLEX]   = UCC_DT_FLOAT128_COMPLEX,
    #else
        [OPAL_DATATYPE_LONG_DOUBLE_COMPLEX]   = COLL_UCC_DT_UNSUPPORTED,
    #endif
#else
    [OPAL_DATATYPE_FLOAT16]                   = COLL_UCC_DT_UNSUPPORTED,
    [OPAL_DATATYPE_FLOAT_COMPLEX]             = COLL_UCC_DT_UNSUPPORTED,
    [OPAL_DATATYPE_DOUBLE_COMPLEX]            = COLL_UCC_DT_UNSUPPORTED,
    [OPAL_DATATYPE_LONG_DOUBLE_COMPLEX]       = COLL_UCC_DT_UNSUPPORTED,
#endif
    [OPAL_DATATYPE_UNAVAILABLE]               = COLL_UCC_DT_UNSUPPORTED
};

#if UCC_HAVE_PAIR_DT
/*
 * The MPI "pair" (MAXLOC/MINLOC) datatypes are OMPI-level composite
 * datatypes (struct/contiguous-of-2 built out of two basic types), not
 * OPAL predefined types, so they cannot be looked up through
 * ompi_datatype_2_ucc_dt[opal_type_id] above (their opal_type_id is
 * either unrelated or, for the block types, aliased onto an OMPI-level
 * id that can collide with an unrelated OPAL id). Look them up by the
 * OMPI-level predefined id (dtype->id) instead, which is stable for the
 * lifetime of the predefined datatype.
 */
static inline ucc_datatype_t ompi_pair_dtype_to_ucc_dtype(int ompi_type_id)
{
    switch (ompi_type_id) {
    case OMPI_DATATYPE_MPI_FLOAT_INT:
        return UCC_DT_FLOAT32_INT;
    case OMPI_DATATYPE_MPI_DOUBLE_INT:
        return UCC_DT_FLOAT64_INT;
    case OMPI_DATATYPE_MPI_LONG_DOUBLE_INT:
#if SIZEOF_LONG_DOUBLE == 16
        return UCC_DT_FLOAT128_INT;
#else
        return COLL_UCC_DT_UNSUPPORTED;
#endif
    case OMPI_DATATYPE_MPI_SHORT_INT:
        return UCC_DT_INT16_INT;
    case OMPI_DATATYPE_MPI_2INT:
        return UCC_DT_INT32_INT;
    case OMPI_DATATYPE_MPI_LONG_INT:
#if SIZEOF_LONG == 8
        return UCC_DT_INT64_INT;
#elif SIZEOF_LONG == 4
        return UCC_DT_INT32_INT;
#else
        return COLL_UCC_DT_UNSUPPORTED;
#endif
    case OMPI_DATATYPE_MPI_2REAL:
        return UCC_DT_2FLOAT32;
    case OMPI_DATATYPE_MPI_2DBLPREC:
        return UCC_DT_2FLOAT64;
    case OMPI_DATATYPE_MPI_2INTEGER:
        return UCC_DT_2INT32;
    default:
        return COLL_UCC_DT_UNSUPPORTED;
    }
}
#endif /* UCC_HAVE_PAIR_DT */

static inline ucc_datatype_t ompi_dtype_to_ucc_dtype(ompi_datatype_t *dtype)
{
    int ompi_type_id = dtype->id;
    int opal_type_id = dtype->super.id;

    if (ompi_type_id < OMPI_DATATYPE_MPI_MAX_PREDEFINED &&
        dtype->super.flags & OMPI_DATATYPE_FLAG_PREDEFINED) {
#if UCC_HAVE_PAIR_DT
        ucc_datatype_t pair_dt = ompi_pair_dtype_to_ucc_dtype(ompi_type_id);
        if (pair_dt != COLL_UCC_DT_UNSUPPORTED) {
            return pair_dt;
        }
#endif
        if (opal_type_id > 0 && opal_type_id < OPAL_DATATYPE_MAX_PREDEFINED) {
            return  ompi_datatype_2_ucc_dt[opal_type_id];
        }
    }
    return COLL_UCC_DT_UNSUPPORTED;
}

/**
 * Map an MPI datatype used by a data movement collective onto a UCC datatype:
 * a predefined type when there is one, otherwise the cached generic datatype
 * backing the MPI derived type. Returns COLL_UCC_DT_UNSUPPORTED when the
 * caller must fall back.
 */
static inline ucc_datatype_t mca_coll_ucc_dtype_get(ompi_datatype_t *dtype)
{
    ucc_datatype_t ucc_dt = ompi_dtype_to_ucc_dtype(dtype);

    if (OPAL_UNLIKELY(COLL_UCC_DT_UNSUPPORTED == ucc_dt)) {
        (void)mca_coll_ucc_derived_dt_get(dtype, &ucc_dt);
    }
    return ucc_dt;
}

/**
 * Check the memory layout of a buffer described by (dtype, count).
 *
 * A derived MPI datatype is offloaded as a UCC generic datatype that carries
 * its own layout, so the contiguity gate must not be applied to it. The gate
 * depends on the local count, which differs between the root and the other
 * ranks of a rooted collective, so applying it to a derived datatype makes the
 * root fall back while the other ranks enter UCC, and the collective hangs.
 */
static inline bool mca_coll_ucc_layout_ok(ompi_datatype_t *dtype, size_t count)
{
    if (COLL_UCC_DT_UNSUPPORTED == ompi_dtype_to_ucc_dtype(dtype)) {
        /* Not a predefined type: either offloaded as a generic datatype, whose
           layout UCC handles, or rejected by mca_coll_ucc_dtype_get() below. */
        return true;
    }
    return ompi_datatype_is_contiguous_memory_layout(dtype, count);
}

static ucc_reduction_op_t ompi_op_to_ucc_op_map[OMPI_OP_BASE_FORTRAN_OP_MAX + 1] = {
   COLL_UCC_OP_UNSUPPORTED,     /* OMPI_OP_BASE_FORTRAN_NULL = 0 */
   UCC_OP_MAX,                  /* OMPI_OP_BASE_FORTRAN_MAX */
   UCC_OP_MIN,                  /* OMPI_OP_BASE_FORTRAN_MIN */
   UCC_OP_SUM,                  /* OMPI_OP_BASE_FORTRAN_SUM */
   UCC_OP_PROD,                 /* OMPI_OP_BASE_FORTRAN_PROD */
   UCC_OP_LAND,                 /* OMPI_OP_BASE_FORTRAN_LAND */
   UCC_OP_BAND,                 /* OMPI_OP_BASE_FORTRAN_BAND */
   UCC_OP_LOR,                  /* OMPI_OP_BASE_FORTRAN_LOR */
   UCC_OP_BOR,                  /* OMPI_OP_BASE_FORTRAN_BOR */
   UCC_OP_LXOR,                 /* OMPI_OP_BASE_FORTRAN_LXOR */
   UCC_OP_BXOR,                 /* OMPI_OP_BASE_FORTRAN_BXOR */
#if UCC_HAVE_PAIR_DT
   UCC_OP_MAXLOC,               /* OMPI_OP_BASE_FORTRAN_MAXLOC */
   UCC_OP_MINLOC,               /* OMPI_OP_BASE_FORTRAN_MINLOC */
#else
   COLL_UCC_OP_UNSUPPORTED,     /* OMPI_OP_BASE_FORTRAN_MAXLOC */
   COLL_UCC_OP_UNSUPPORTED,     /* OMPI_OP_BASE_FORTRAN_MINLOC */
#endif
   COLL_UCC_OP_UNSUPPORTED,     /* OMPI_OP_BASE_FORTRAN_REPLACE */
   COLL_UCC_OP_UNSUPPORTED,     /* OMPI_OP_BASE_FORTRAN_NO_OP */
   COLL_UCC_OP_UNSUPPORTED      /* OMPI_OP_BASE_FORTRAN_OP_MAX */
};

static inline ucc_reduction_op_t ompi_op_to_ucc_op(ompi_op_t *op) {
    if (op->o_f_to_c_index > OMPI_OP_BASE_FORTRAN_OP_MAX) {
        return COLL_UCC_OP_UNSUPPORTED;
    }
    return ompi_op_to_ucc_op_map[op->o_f_to_c_index];
}

#endif /* COLL_UCC_DTYPES_H */
