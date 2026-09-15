/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil -*- */
/*
 * Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
 * $COPYRIGHT$
 *
 * Additional copyrights may follow
 *
 * $HEADER$
 */

#include "ompi_config.h"
#include "coll_ucc.h"
#include "coll_ucc_dtypes.h"
#include "opal/class/opal_list.h"
#include "opal/mca/threads/mutex.h"

#if UCC_HAVE_GENERIC_DT_REDUCE

/**
 * Cache entry: a UCC generic datatype that carries the user-defined MPI
 * reduction operation for one (op, dtype) pair. The MPI op and datatype are
 * retained for the lifetime of the entry so that the callback can use them
 * even if the application frees its handles. Entries live until the UCC
 * component tears down the UCC context.
 */
typedef struct mca_coll_ucc_user_op_t {
    opal_list_item_t         super;
    struct ompi_op_t        *op;
    struct ompi_datatype_t  *dtype;
    size_t                   extent;
    ucc_datatype_t           ucc_dt;
} mca_coll_ucc_user_op_t;

static OBJ_CLASS_INSTANCE(mca_coll_ucc_user_op_t, opal_list_item_t, NULL, NULL);

/**
 * Find the cache entry a reduce callback belongs to. ucc_dt_generic_t is
 * opaque in the UCC public API and ucc_reduce_cb_params_t.cb_ctx is not set
 * by UCC, so the datatype object address - which is the payload of the
 * ucc_datatype_t handle - is used as the key.
 */
static mca_coll_ucc_user_op_t *
mca_coll_ucc_user_op_find(const ucc_dt_generic_t *dt)
{
    mca_coll_ucc_component_t *cm  = &mca_coll_ucc_component;
    uint64_t                  key = (uint64_t)(uintptr_t)dt;
    mca_coll_ucc_user_op_t   *uop, *found = NULL;

    opal_mutex_lock(&cm->user_ops_lock);
    OPAL_LIST_FOREACH(uop, &cm->user_ops, mca_coll_ucc_user_op_t) {
        if ((uop->ucc_dt & ~(uint64_t)UCC_DATATYPE_CLASS_MASK) == key) {
            found = uop;
            break;
        }
    }
    opal_mutex_unlock(&cm->user_ops_lock);
    return found;
}

/**
 * UCC user-defined reduction callback.
 *
 * UCC uses two layouts for the input vectors (see ucc_dt_reduce.h):
 *  - strided: inputs are "src1" plus "n_vectors" buffers starting at "src2",
 *    "stride" bytes apart;
 *  - multi:   "src2" is NULL and "src1" is an array of "n_vectors" buffer
 *    pointers.
 * In both cases each input holds "count" elements of the generic datatype and
 * the result must be stored in "dst", which is allowed to alias any input.
 * The reduction op of the collective (ucc_coll_args_t.op) does not apply and
 * neither does alpha: the operation is fully defined by this callback.
 */
static ucc_status_t mca_coll_ucc_reduce_cb(const ucc_reduce_cb_params_t *params)
{
    mca_coll_ucc_user_op_t *uop      = mca_coll_ucc_user_op_find(params->dt);
    void  **srcs_ext = (NULL == params->src2) ? (void **)params->src1 : NULL;
    size_t  n_srcs   = (NULL == params->src2) ? params->n_vectors
                                              : params->n_vectors + 1;
    void   *dst      = params->dst;
    size_t  i;
    ssize_t alias;
    void   *src;

#define COLL_UCC_REDUCE_CB_SRC(_i)                                       \
    ((NULL != srcs_ext) ? srcs_ext[(_i)]                                 \
                        : ((0 == (_i)) ? params->src1                    \
                                       : (void *)((char *)params->src2 + \
                                                  ((_i) - 1) * params->stride)))

    if (OPAL_UNLIKELY(NULL == uop)) {
        UCC_ERROR("user op reduce callback on an unknown datatype");
        return UCC_ERR_INVALID_PARAM;
    }
    if (0 == n_srcs || 0 == params->count) {
        return UCC_OK;
    }

    /* The result buffer may alias one of the inputs: in that case the
     * remaining inputs are accumulated into it, otherwise it is seeded with
     * the first input. The op is commutative (enforced at selection time), so
     * the accumulation order is irrelevant. */
    alias = -1;
    for (i = 0; i < n_srcs; i++) {
        if (COLL_UCC_REDUCE_CB_SRC(i) == dst) {
            alias = (ssize_t)i;
            break;
        }
    }

    if (alias < 0) {
        memcpy(dst, COLL_UCC_REDUCE_CB_SRC(0), params->count * uop->extent);
        alias = 0;
    }

    for (i = 0; i < n_srcs; i++) {
        if ((ssize_t)i == alias) {
            continue;
        }
        src = COLL_UCC_REDUCE_CB_SRC(i);
        ompi_op_reduce(uop->op, src, dst, params->count, uop->dtype);
    }
#undef COLL_UCC_REDUCE_CB_SRC

    return UCC_OK;
}

static ucc_datatype_t mca_coll_ucc_user_op_create(struct ompi_op_t *op,
                                                  struct ompi_datatype_t *dtype,
                                                  size_t extent)
{
    mca_coll_ucc_user_op_t *uop;
    ucc_generic_dt_ops_t    ops;
    ucc_status_t            status;

    uop = OBJ_NEW(mca_coll_ucc_user_op_t);
    if (NULL == uop) {
        return COLL_UCC_DT_UNSUPPORTED;
    }
    uop->op     = op;
    uop->dtype  = dtype;
    uop->extent = extent;

    memset(&ops, 0, sizeof(ops));
    ops.mask        = UCC_GENERIC_DT_OPS_FIELD_FLAGS;
    ops.flags       = UCC_GENERIC_DT_OPS_FLAG_CONTIG |
                      UCC_GENERIC_DT_OPS_FLAG_REDUCE;
    ops.contig_size = extent;
    /* pack/unpack are left unset: UCC only uses them for non-contiguous
     * generic datatypes, and this one is contiguous. */
    ops.reduce.cb     = mca_coll_ucc_reduce_cb;
    /* cb_ctx is not propagated to the callback by UCC; the datatype object
     * address is used to recover the entry instead (see the find helper). */
    ops.reduce.cb_ctx = uop;

    status = ucc_dt_create_generic(&ops, uop, &uop->ucc_dt);
    if (UCC_OK != status) {
        UCC_VERBOSE(5, "failed to create ucc generic dt for op %s: %s",
                    op->o_name, ucc_status_string(status));
        OBJ_RELEASE(uop);
        return COLL_UCC_DT_UNSUPPORTED;
    }

    OBJ_RETAIN(op);
    OBJ_RETAIN(dtype);
    opal_list_append(&mca_coll_ucc_component.user_ops, &uop->super);
    UCC_VERBOSE(5, "created ucc generic dt for user op %s, dtype %s",
                op->o_name, dtype->super.name);
    return uop->ucc_dt;
}

ucc_datatype_t mca_coll_ucc_user_op_dtype(struct ompi_op_t *op,
                                          struct ompi_datatype_t *dtype)
{
    mca_coll_ucc_component_t *cm = &mca_coll_ucc_component;
    mca_coll_ucc_user_op_t   *uop;
    ucc_datatype_t            ucc_dt;
    size_t                    extent;

    if (!cm->ucc_user_ops_enable || ompi_op_is_intrinsic(op)) {
        return COLL_UCC_DT_UNSUPPORTED;
    }
    if (!ompi_op_is_commute(op)) {
        UCC_VERBOSE(5, "non-commutative user op is not supported: op = %s",
                    op->o_name);
        return COLL_UCC_DT_UNSUPPORTED;
    }
    if (!mca_coll_ucc_dtype_is_dense(dtype, &extent)) {
        UCC_VERBOSE(5, "user op with non-contiguous dtype is not supported: "
                    "dtype = %s", dtype->super.name);
        return COLL_UCC_DT_UNSUPPORTED;
    }

    ucc_dt = COLL_UCC_DT_UNSUPPORTED;
    opal_mutex_lock(&cm->user_ops_lock);
    OPAL_LIST_FOREACH(uop, &cm->user_ops, mca_coll_ucc_user_op_t) {
        if (uop->op == op && uop->dtype == dtype) {
            ucc_dt = uop->ucc_dt;
            break;
        }
    }
    if (COLL_UCC_DT_UNSUPPORTED == ucc_dt) {
        ucc_dt = mca_coll_ucc_user_op_create(op, dtype, extent);
    }
    opal_mutex_unlock(&cm->user_ops_lock);

    return ucc_dt;
}

void mca_coll_ucc_user_ops_cleanup(void)
{
    mca_coll_ucc_component_t *cm = &mca_coll_ucc_component;
    mca_coll_ucc_user_op_t   *uop;
    opal_list_item_t         *item;

    opal_mutex_lock(&cm->user_ops_lock);
    while (NULL != (item = opal_list_remove_first(&cm->user_ops))) {
        uop = (mca_coll_ucc_user_op_t *)item;
        ucc_dt_destroy(uop->ucc_dt);
        OBJ_RELEASE(uop->op);
        OBJ_RELEASE(uop->dtype);
        OBJ_RELEASE(uop);
    }
    opal_mutex_unlock(&cm->user_ops_lock);
}

#else /* !UCC_HAVE_GENERIC_DT_REDUCE */

ucc_datatype_t mca_coll_ucc_user_op_dtype(struct ompi_op_t *op,
                                          struct ompi_datatype_t *dtype)
{
    (void)op;
    (void)dtype;
    return COLL_UCC_DT_UNSUPPORTED;
}

void mca_coll_ucc_user_ops_cleanup(void)
{
}

#endif /* UCC_HAVE_GENERIC_DT_REDUCE */

ucc_status_t mca_coll_ucc_map_reduce_op(struct ompi_datatype_t *dtype,
                                        struct ompi_op_t *op,
                                        ucc_datatype_t *ucc_dt,
                                        ucc_reduction_op_t *ucc_op)
{
    ucc_datatype_t user_dt;

    *ucc_dt = ompi_dtype_to_ucc_dtype(dtype);
    *ucc_op = ompi_op_to_ucc_op(op);

    if (OPAL_UNLIKELY(COLL_UCC_OP_UNSUPPORTED == *ucc_op)) {
        user_dt = mca_coll_ucc_user_op_dtype(op, dtype);
        if (COLL_UCC_DT_UNSUPPORTED != user_dt) {
            /* The reduction is carried by the datatype callback; UCC ignores
             * coll_args.op for generic datatypes, but it must not be AVG
             * because some algorithms use it to trigger post-processing. */
            *ucc_dt = user_dt;
            *ucc_op = UCC_OP_SUM;
            return UCC_OK;
        }
        UCC_VERBOSE(5, "ompi_op is not supported: op = %s", op->o_name);
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (OPAL_UNLIKELY(COLL_UCC_DT_UNSUPPORTED == *ucc_dt)) {
        /* Note: a derived datatype never reaches this point with an
         * intrinsic op - MPI only defines the predefined ops on the
         * predefined datatypes, and ompi_op_is_valid() rejects the call
         * before the collective module is invoked. */
        UCC_VERBOSE(5, "ompi_datatype is not supported: dtype = %s",
                    dtype->super.name);
        return UCC_ERR_NOT_SUPPORTED;
    }
    return UCC_OK;
}
