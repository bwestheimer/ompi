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
#include "ompi/datatype/ompi_datatype.h"
#include "opal/datatype/opal_convertor.h"
#include "opal/class/opal_list.h"
#include "opal/mca/threads/mutex.h"

/**
 * A derived MPI datatype is "dense" when "count" elements of it occupy exactly
 * "count * size" bytes starting at the buffer address the user passes in. Such
 * a datatype can be handed to UCC as a contiguous generic datatype: UCC then
 * moves it with plain byte copies and computes every displacement as a
 * multiple of contig_size, exactly like it does for a predefined datatype.
 *
 * A non-zero lower bound is rejected on purpose. UCC has no notion of a buffer
 * base offset: the element displacements it derives (and the displacement
 * arrays of the v-variants, which are expressed in datatype units) are all
 * relative to the buffer pointer, so a shifted type would need every buffer
 * and every displacement of the collective to be adjusted, which is not
 * expressible for the v-collectives.
 */
int mca_coll_ucc_dtype_is_dense(struct ompi_datatype_t *dtype,
                                size_t *extent_out)
{
    ptrdiff_t extent, lb, true_extent, true_lb;
    size_t    size;

    if (!(dtype->super.flags & OPAL_DATATYPE_FLAG_CONTIGUOUS)) {
        return 0;
    }
    if (OMPI_SUCCESS != ompi_datatype_type_size(dtype, &size) ||
        OMPI_SUCCESS != ompi_datatype_get_extent(dtype, &lb, &extent) ||
        OMPI_SUCCESS != ompi_datatype_get_true_extent(dtype, &true_lb,
                                                      &true_extent)) {
        return 0;
    }
    if (0 == size || 0 != lb || 0 != true_lb || extent != (ptrdiff_t)size ||
        true_extent != extent) {
        return 0;
    }
    *extent_out = (size_t)extent;
    return 1;
}

#if UCC_HAVE_GENERIC_DT

/**
 * Cache entry: one UCC generic datatype per MPI derived datatype. The MPI
 * datatype is retained for the lifetime of the entry, which both keeps the
 * OPAL convertor description alive and guarantees that the datatype address
 * used as the cache key cannot be recycled by a later MPI_Type_commit().
 * Entries live until the UCC context is torn down.
 */
typedef struct mca_coll_ucc_derived_dt_t {
    opal_list_item_t         super;
    struct ompi_datatype_t  *dtype;
    ucc_datatype_t           ucc_dt;
} mca_coll_ucc_derived_dt_t;

static OBJ_CLASS_INSTANCE(mca_coll_ucc_derived_dt_t, opal_list_item_t, NULL,
                          NULL);

/**
 * Pack/unpack state handed back to UCC by start_pack()/start_unpack() and
 * passed to every subsequent pack()/unpack()/finish() call.
 */
typedef struct mca_coll_ucc_dt_conv_t {
    opal_convertor_t         conv;
    struct ompi_datatype_t  *dtype;
} mca_coll_ucc_dt_conv_t;

static mca_coll_ucc_dt_conv_t *
mca_coll_ucc_dt_conv_get(void *context, void *buffer, size_t count, int send)
{
    struct ompi_datatype_t *dtype = (struct ompi_datatype_t *)context;
    mca_coll_ucc_dt_conv_t *state;

    state = (mca_coll_ucc_dt_conv_t *)malloc(sizeof(*state));
    if (OPAL_UNLIKELY(NULL == state)) {
        UCC_ERROR("failed to allocate a convertor for datatype %s",
                  dtype->super.name);
        return NULL;
    }
    state->dtype = dtype;
    OBJ_CONSTRUCT(&state->conv, opal_convertor_t);
    OMPI_DATATYPE_RETAIN(dtype);
    if (send) {
        opal_convertor_copy_and_prepare_for_send(ompi_mpi_local_convertor,
                                                 &dtype->super, count, buffer,
                                                 0, &state->conv);
    } else {
        opal_convertor_copy_and_prepare_for_recv(ompi_mpi_local_convertor,
                                                 &dtype->super, count, buffer,
                                                 0, &state->conv);
    }
    return state;
}

static void *mca_coll_ucc_dt_start_pack(void *context, const void *buffer,
                                        size_t count)
{
    return mca_coll_ucc_dt_conv_get(context, (void *)buffer, count, 1);
}

static void *mca_coll_ucc_dt_start_unpack(void *context, void *buffer,
                                          size_t count)
{
    return mca_coll_ucc_dt_conv_get(context, buffer, count, 0);
}

static size_t mca_coll_ucc_dt_packed_size(void *state)
{
    mca_coll_ucc_dt_conv_t *conv_state = (mca_coll_ucc_dt_conv_t *)state;
    size_t                  size;

    opal_convertor_get_packed_size(&conv_state->conv, &size);
    return size;
}

/**
 * "offset" is a byte offset into the packed stream (same contract as the UCP
 * generic datatype used by pml/ucx), so the convertor is repositioned on every
 * call: UCC may pack sub-ranges of the stream, and it may do so out of order.
 */
static size_t mca_coll_ucc_dt_pack(void *state, size_t offset, void *dest,
                                   size_t max_length)
{
    mca_coll_ucc_dt_conv_t *conv_state = (mca_coll_ucc_dt_conv_t *)state;
    uint32_t                iov_count  = 1;
    size_t                  length     = max_length;
    struct iovec            iov;

    iov.iov_base = dest;
    iov.iov_len  = max_length;

    opal_convertor_set_position(&conv_state->conv, &offset);
    opal_convertor_pack(&conv_state->conv, &iov, &iov_count, &length);
    return length;
}

static ucc_status_t mca_coll_ucc_dt_unpack(void *state, size_t offset,
                                           const void *src, size_t length)
{
    mca_coll_ucc_dt_conv_t *conv_state = (mca_coll_ucc_dt_conv_t *)state;
    uint32_t                iov_count  = 1;
    size_t                  len        = length;
    struct iovec            iov;

    iov.iov_base = (void *)src;
    iov.iov_len  = length;

    opal_convertor_set_position(&conv_state->conv, &offset);
    opal_convertor_unpack(&conv_state->conv, &iov, &iov_count, &len);
    return (len == length) ? UCC_OK : UCC_ERR_INVALID_PARAM;
}

static void mca_coll_ucc_dt_finish(void *state)
{
    mca_coll_ucc_dt_conv_t *conv_state = (mca_coll_ucc_dt_conv_t *)state;
    struct ompi_datatype_t *dtype      = conv_state->dtype;

    opal_convertor_cleanup(&conv_state->conv);
    OBJ_DESTRUCT(&conv_state->conv);
    OMPI_DATATYPE_RELEASE(dtype);
    free(conv_state);
}

static ucc_datatype_t
mca_coll_ucc_derived_dt_create(struct ompi_datatype_t *dtype, int is_dense,
                               size_t extent)
{
    mca_coll_ucc_derived_dt_t *ddt;
    ucc_generic_dt_ops_t       ops;
    ucc_status_t               status;

    ddt = OBJ_NEW(mca_coll_ucc_derived_dt_t);
    if (NULL == ddt) {
        return COLL_UCC_DT_UNSUPPORTED;
    }
    ddt->dtype = dtype;

    memset(&ops, 0, sizeof(ops));
    ops.mask         = UCC_GENERIC_DT_OPS_FIELD_FLAGS;
    ops.start_pack   = mca_coll_ucc_dt_start_pack;
    ops.start_unpack = mca_coll_ucc_dt_start_unpack;
    ops.packed_size  = mca_coll_ucc_dt_packed_size;
    ops.pack         = mca_coll_ucc_dt_pack;
    ops.unpack       = mca_coll_ucc_dt_unpack;
    ops.finish       = mca_coll_ucc_dt_finish;
    if (is_dense) {
        /* The pack/unpack callbacks stay installed: they are correct for this
         * datatype as well, UCC just does not need them. */
        ops.flags       = UCC_GENERIC_DT_OPS_FLAG_CONTIG;
        ops.contig_size = extent;
    }

    status = ucc_dt_create_generic(&ops, dtype, &ddt->ucc_dt);
    if (UCC_OK != status) {
        UCC_VERBOSE(5, "failed to create ucc generic dt for %s: %s",
                    dtype->super.name, ucc_status_string(status));
        OBJ_RELEASE(ddt);
        return COLL_UCC_DT_UNSUPPORTED;
    }

    OBJ_RETAIN(dtype);
    opal_list_append(&mca_coll_ucc_component.derived_dts, &ddt->super);
    UCC_VERBOSE(5, "created %s ucc generic dt for derived dtype %s",
                is_dense ? "contiguous" : "packed", dtype->super.name);
    return ddt->ucc_dt;
}

ucc_status_t mca_coll_ucc_derived_dt_get(struct ompi_datatype_t *dtype,
                                         ucc_datatype_t *ucc_dt)
{
    mca_coll_ucc_component_t  *cm = &mca_coll_ucc_component;
    mca_coll_ucc_derived_dt_t *ddt;
    ucc_datatype_t             dt;
    size_t                     extent;
    int                        is_dense;

    *ucc_dt = COLL_UCC_DT_UNSUPPORTED;
    if (0 == cm->ucc_derived_dt_enable) {
        return UCC_ERR_NOT_SUPPORTED;
    }
    /* MPI_DATATYPE_NULL and friends have no usable description. */
    if (NULL == dtype || 0 == dtype->super.size) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    is_dense = mca_coll_ucc_dtype_is_dense(dtype, &extent);
    if (!is_dense && cm->ucc_derived_dt_enable < 2) {
        UCC_VERBOSE(5, "non-contiguous dtype is not supported: dtype = %s",
                    dtype->super.name);
        return UCC_ERR_NOT_SUPPORTED;
    }

    dt = COLL_UCC_DT_UNSUPPORTED;
    opal_mutex_lock(&cm->derived_dts_lock);
    OPAL_LIST_FOREACH(ddt, &cm->derived_dts, mca_coll_ucc_derived_dt_t) {
        if (ddt->dtype == dtype) {
            dt = ddt->ucc_dt;
            break;
        }
    }
    if (COLL_UCC_DT_UNSUPPORTED == dt) {
        dt = mca_coll_ucc_derived_dt_create(dtype, is_dense, extent);
    }
    opal_mutex_unlock(&cm->derived_dts_lock);

    if (COLL_UCC_DT_UNSUPPORTED == dt) {
        return UCC_ERR_NOT_SUPPORTED;
    }
    *ucc_dt = dt;
    return UCC_OK;
}

void mca_coll_ucc_derived_dts_cleanup(void)
{
    mca_coll_ucc_component_t  *cm = &mca_coll_ucc_component;
    mca_coll_ucc_derived_dt_t *ddt;
    opal_list_item_t          *item;

    opal_mutex_lock(&cm->derived_dts_lock);
    while (NULL != (item = opal_list_remove_first(&cm->derived_dts))) {
        ddt = (mca_coll_ucc_derived_dt_t *)item;
        ucc_dt_destroy(ddt->ucc_dt);
        OBJ_RELEASE(ddt->dtype);
        OBJ_RELEASE(ddt);
    }
    opal_mutex_unlock(&cm->derived_dts_lock);
}

#else /* !UCC_HAVE_GENERIC_DT */

ucc_status_t mca_coll_ucc_derived_dt_get(struct ompi_datatype_t *dtype,
                                         ucc_datatype_t *ucc_dt)
{
    (void)dtype;
    *ucc_dt = COLL_UCC_DT_UNSUPPORTED;
    return UCC_ERR_NOT_SUPPORTED;
}

void mca_coll_ucc_derived_dts_cleanup(void)
{
}

#endif /* UCC_HAVE_GENERIC_DT */
