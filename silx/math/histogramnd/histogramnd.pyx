# /*##########################################################################
# coding: utf-8
# Copyright (C) 2016 European Synchrotron Radiation Facility
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# ############################################################################*/

__authors__ = ["D. Naudet"]
__license__ = "MIT"
__date__ = "01/02/2016"

cimport numpy
cimport cython
import numpy

cdef enum _opt_type:
    HISTO_NONE              = 0
    HISTO_WEIGHT_MIN        = 1,       # Filter weights with i_weight_min.
    HISTO_WEIGHT_MAX        = 1 << 1,  # Filter weights with i_weight_max.
    HISTO_LAST_BIN_CLOSED   = 1 << 2   # Last bin is closed.

ctypedef fused sample_t:
    numpy.float64_t
    numpy.float32_t
    numpy.int32_t
    numpy.int64_t

ctypedef fused weights_t:
    numpy.float64_t
    numpy.float32_t
    numpy.int32_t
    numpy.int64_t


def histogramnd(sample,
                bins_rng,
                n_bins,
                weights=None,
                weight_min=None,
                weight_max=None,
                last_bin_closed=False,
                histo=None,
                cumul=None):
    """
    histogramnd(sample, weights, bins_rng, n_bins, weight_min=None, weight_max=None, last_bin_closed=False, histo=None, cumul=None)

    Computes the multidimensional histogram of some data.

    :param sample:
        The data to be histogrammed.
        Its shape must be either
        (N,) if it contains one dimensional coordinates,
        or an (N,D) array where the rows are the
        coordinates of points in a D dimensional space.
        The following dtypes are supported : :class:`numpy.float64`,
        :class:`numpy.float32`, :class:`numpy.int32`.
    :type sample: :class:`numpy.array`

    :param bins_rng:
        A (N, 2) array containing the lower and upper
        bin edges along each dimension.
    :type bins_rng: array_like

    :param n_bins:
        The number of bins :
            * a scalar (same number of bins for all dimensions)
            * a D elements array (number of bins for each dimensions)
    :type n_bins: scalar or array_like

    :param weights:
        A N elements numpy array of values associated with
        each sample.
        The values of the *cumul* array
        returned by the function are equal to the sum of
        the weights associated with the samples falling
        into each bin.
        The following dtypes are supported : :class:`numpy.float64`,
        :class:`numpy.float32`, :class:`numpy.int32`.

        .. note:: If None, the weighted histogram returned will be None.
    :type weights: *optional*, :class:`numpy.array`

    :param weight_min:
        Use this parameter to filter out all samples whose
        weights are lower than this value.

        .. note:: This value will be cast to the same type
            as *weights*.
    :type weight_min: *optional*, scalar

    :param weight_max:
        Use this parameter to filter out all samples whose
        weights are higher than this value.

        .. note:: This value will be cast to the same type
            as *weights*.

    :type weight_max: *optional*, scalar

    :param last_bin_closed:
        By default the last bin is half
        open (i.e.: [x,y) ; x included, y
        excluded), like all the other bins.
        Set this parameter to true if you want
        the LAST bin to be closed.
    :type last_bin_closed: *optional*, :class:`python.boolean`

    :param histo:
        Use this parameter if you want to pass your
        own histogram array instead of the one
        created by this function. New values
        will be added to this array. The returned array
        will then be this one (same reference).

        .. warning:: If the histo array was created by a previous
            call to histogramnd then the user is
            responsible for providing the same parameters
            (*n_bins*, *bins_rng*, ...).
    :type histo: *optional*, :class:`numpy.array`

    :param cumul:
        Use this parameter if you want to pass your
        own weighted histogram array instead of
        the created by this function. New
        values will be added to this array. The returned array
        will then be this one (same reference).

        .. warning:: If the cumul array was created by a previous
            call to histogramnd then the user is
            responsible for providing the same parameters
            (*n_bins*, *bins_rng*, ...).
    :type cumul: *optional*, :class:`numpy.array`

    :return: Histogram (bin counts, always returned) and weighted histogram of
        the sample (or *None* if weights is *None*).
    :rtype: *tuple* (:class:`numpy.array`, :class:`numpy.array`) or
        (:class:`numpy.array`, None)
    """

    s_shape = sample.shape

    n_dims = 1 if len(s_shape) == 1 else s_shape[1]

    if weights is not None:
        w_shape = weights.shape

        # making sure the sample and weights sizes are coherent
        # 2 different cases : 2D sample (N,M) and 1D (N)
        if len(w_shape) != 1 or w_shape[0] != s_shape[0]:
            raise ValueError('<weights> must be an array whose length '
                             'is equal to the number of samples.')

        weights_type = weights.dtype
    else:
        weights_type = None

    # just in case those arent numpy arrays
    # (this allows the user to provide native python lists,
    #   => easier for testing)
    i_bins_rng = bins_rng
    bins_rng = numpy.array(bins_rng)
    err_bins_rng = False

    if n_dims == 1:
        if bins_rng.shape == (2,):
            pass
        elif bins_rng.shape == (1, 2):
            bins_rng.shape = -1
        else:
            err_bins_rng = True
    elif n_dims != 1 and bins_rng.shape != (n_dims, 2):
        err_bins_rng = True

    if err_bins_rng:
        raise ValueError('<bins_rng> error : expected {n_dims} sets of '
                         'lower and upper bin edges, '
                         'got the following instead : {bins_rng}. '
                         '(provided <sample> contains '
                         '{n_dims}D values)'
                         ''.format(bins_rng=i_bins_rng,
                                   n_dims=n_dims))

    # checking n_bins size
    n_bins = numpy.array(n_bins, ndmin=1)
    if len(n_bins) == 1:
        n_bins = numpy.tile(n_bins, n_dims)
    elif n_bins.shape != (n_dims,):
        raise ValueError('n_bins must be either a scalar (same number '
                         'of bins for all dimensions) or '
                         'an array (number of bins for each '
                         'dimension).')

    # checking if None is in n_bins, otherwise a rather cryptic
    #   exception is thrown when calling numpy.zeros
    # also testing for negative/null values
    if numpy.any(numpy.equal(n_bins, None)) or numpy.any(n_bins <= 0):
        raise ValueError('<n_bins> : only positive values allowed.')

    output_shape = tuple(n_bins)

    # checking the histo array, if provided
    if histo is None:
        histo = numpy.zeros(output_shape, dtype=numpy.uint32)
    else:
        if histo.shape != output_shape:
            raise ValueError('Provided <histo> array doesn\'t have '
                             'a shape compatible with <n_bins> '
                             ': should be {0} instead of {1}.'
                             ''.format(output_shape, histo.shape))
        if histo.dtype != numpy.uint32:
            raise ValueError('Provided <histo> array doesn\'t have '
                             'the expected type '
                             ': should be {0} instead of {1}.'
                             ''.format(numpy.uint32, histo.dtype))

    # checking the cumul array, if provided
    if weights_type is None:
        cumul = None
    elif cumul is None:
        cumul = numpy.zeros(output_shape, dtype=numpy.double)
    else:
        if cumul.shape != output_shape:
            raise ValueError('Provided <cumul> array doesn\'t have '
                             'a shape compatible with <n_bins> '
                             ': should be {0} instead of {1}.'
                             ''.format(output_shape, cumul.shape))
        if cumul.dtype != numpy.float:
            raise ValueError('Provided <cumul> array doesn\'t have '
                             'the expected type '
                             ': should be {0} instead of {1}.'
                             ''.format(numpy.double, cumul.dtype))

    option_flags = 0

    if weight_min is not None:
        option_flags |= HISTO_WEIGHT_MIN
    else:
        weight_min = 0

    if weight_max is not None:
        option_flags |= HISTO_WEIGHT_MAX
    else:
        weight_max = 0

    if last_bin_closed is not None and last_bin_closed:
        option_flags |= HISTO_LAST_BIN_CLOSED

    sample_type = sample.dtype

    n_elem = sample.size // n_dims

    # wanted to store the functions in a dict (with the supported types
    # as keys, but I couldn't find a way to make it work with cdef
    # functions. so I have to explicitly list them all...

    def raise_unsupported_type():
        raise TypeError('Case not supported - sample:{0} '
                        'and weights:{1}.'
                        ''.format(sample_type, weights_type))

    sample_c = numpy.ascontiguousarray(sample.reshape((sample.size,)))

    weights_c = (numpy.ascontiguousarray(weights.reshape((weights.size,)))
                 if weights is not None else None)

    bins_rng_c = numpy.ascontiguousarray(bins_rng.reshape((bins_rng.size,)),
                                         dtype=sample_type)

    n_bins_c = numpy.ascontiguousarray(n_bins.reshape((n_bins.size,)),
                                       dtype=numpy.int32)

    histo_c = numpy.ascontiguousarray(histo.reshape((histo.size,)))

    if cumul is not None:
        cumul_c = numpy.ascontiguousarray(cumul.reshape((cumul.size,)))
    else:
        cumul_c = None

    try:
        histo_rc = _histogramnd_fused(sample_c,
                                      weights_c,
                                      n_dims,
                                      n_elem,
                                      bins_rng_c,
                                      n_bins_c,
                                      histo_c,
                                      cumul_c,
                                      option_flags,
                                      weight_min,
                                      weight_max)
    except TypeError as ex:
        raise_unsupported_type()

    if histo_rc != 0:
            raise Exception('histogramnd returned an error : {0}'
                            ''.format(histo_rc))

    return histo, cumul


# =====================
# =====================


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def _histogramnd_fused(sample_t[:] i_sample,
                       weights_t[:] i_weights,
                       int i_n_dims,
                       int i_n_elems,
                       sample_t[:] i_bins_rng,
                       int[:] i_n_bins,
                       numpy.uint32_t[:] o_histo,
                       numpy.float64_t[:] o_cumul,
                       int i_opt_flags,
                       weights_t i_weight_min,
                       weights_t i_weight_max):

    cdef:
        int i = 0
        long elem_idx = 0
        long max_idx = 0
        long weight_idx = 0

        # computed bin index (i_sample -> grid)
        long bin_idx = 0

        sample_t elem_coord = 0

        sample_t[50] g_min
        sample_t[50] g_max
        sample_t[50] bins_range

        int filt_min_weight = 0
        int filt_max_weight = 0
        int last_bin_closed = 0

    for i in range(i_n_dims):
        g_min[i] = i_bins_rng[2*i]
        g_max[i] = i_bins_rng[2*i+1]
        bins_range[i] = g_max[i] - g_min[i]

    # Testing the option flags
    if i_opt_flags & HISTO_WEIGHT_MIN:
        filt_min_weight = 1

    if i_opt_flags & HISTO_WEIGHT_MAX:
        filt_max_weight = 1

    if i_opt_flags & HISTO_LAST_BIN_CLOSED:
        last_bin_closed = 1

    if i_weights is None:
        # if weights are not provided there no point in trying to filter them
        # (!! careful if you change this, some code below relies on it !!)
        filt_min_weight = 0
        filt_max_weight = 0

        # If the weights array is not provided then there is no point
        # updating the weighted histogram, only the bin counts (o_histo)
        # will be filled.
        # (!! careful if you change this, some code below relies on it !!)
        o_cumul = None

    weight_idx = -1
    elem_idx = 0 - i_n_dims
    max_idx = i_n_elems * i_n_dims - i_n_dims

    with nogil:
        while elem_idx < max_idx:
            elem_idx += i_n_dims
            weight_idx += 1

            if filt_min_weight and i_weights[weight_idx] < i_weight_min:
                continue
            if filt_max_weight and i_weights[weight_idx] > i_weight_max:
                continue

            bin_idx = 0

            for i in range(i_n_dims):
                elem_coord = i_sample[elem_idx+i]
                # =====================
                # Element is rejected if any of the following is NOT true :
                # 1. coordinate is >= than the minimum value
                # 2. coordinate is <= than the maximum value
                # 3. coordinate==maximum value and last_bin_closed is True
                # =====================
                if elem_coord < g_min[i]:
                    bin_idx = -1
                    break

                # Here we make the assumption that most of the time
                # there will be more coordinates inside the grid interval
                #  (one test)
                #  than coordinates higher or equal to the max
                #  (two tests)
                if elem_coord < g_max[i]:
                    bin_idx = <long>(bin_idx * i_n_bins[i] +
                                     (((elem_coord - g_min[i]) * i_n_bins[i]) /
                                      bins_range[i]))
                else:
                    # if equal and the last bin is closed :
                    #  put it in the last bin
                    # else : discard
                    if last_bin_closed and elem_coord == g_max[i]:
                        bin_idx = (bin_idx + 1) * i_n_bins[i] - 1
                    else:
                        bin_idx = -1
                        break

            # element is out of the grid
            if bin_idx == -1:
                continue

            if o_histo is not None:
                o_histo[bin_idx] += 1

            if o_cumul is not None:
                # not testing the pointer since o_cumul is null if
                # i_weights is null.
                o_cumul[bin_idx] += i_weights[weight_idx]

    return 0
