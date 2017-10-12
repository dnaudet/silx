# -*- coding: utf-8 -*-
#
#    Project: silx
#             https://github.com/silx-kit/silx
#
#    Copyright (C) 2012-2017 European Synchrotron Radiation Facility, Grenoble, France
#
#    Principal author:       Jérôme Kieffer (Jerome.Kieffer@ESRF.eu)
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#  .
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#  .
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#  THE SOFTWARE.

"""A general purpose library for manipulating 2D images in 1 or 3 colors 

"""
from __future__ import absolute_import, print_function, with_statement, division


__author__ = "Jerome Kieffer"
__license__ = "MIT"
__date__ = "12/10/2017"
__copyright__ = "2012-2017, ESRF, Grenoble"
__contact__ = "jerome.kieffer@esrf.fr"

import os
import logging
import numpy
from collections import OrderedDict
from math import floor, sqrt, log

from .common import pyopencl, kernel_workgroup_size
from .processing import EventDescription, OpenclProcessing, BufferDescription

if pyopencl:
    mf = pyopencl.mem_flags
else:
    raise ImportError("pyopencl is not installed")
logger = logging.getLogger(__name__)


class ImageProcessing(OpenclProcessing):

    kernel_files = ["cast", "map", "max_min"]

    converter = {numpy.dtype(numpy.uint8): "u8_to_float",
                 numpy.dtype(numpy.int8): "s8_to_float",
                 numpy.dtype(numpy.uint16): "u16_to_float",
                 numpy.dtype(numpy.int16): "s16_to_float",
                 numpy.dtype(numpy.uint32): "u32_to_float",
                 numpy.dtype(numpy.int32): "s32_to_float",
                 }

    def __init__(self, shape=None, ncolors=1, template=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, memory=None, profile=False):
        """Constructor of the ImageProcessing class

        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the
                            out come of the compilation
        :param memory: minimum memory available on device
        :param profile: switch on profiling to be able to profile at the kernel
                         level, store profiling elements (makes code slightly slower)
        """
        OpenclProcessing.__init__(self, ctx=ctx, devicetype=devicetype,
                                  platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, memory=memory, profile=profile)
        if template is not None:
            shape = template.shape
            if len(shape) > 2:
                self.ncolors = shape[2]
                self.shape = shape[:2]
            else:
                self.ncolors = 1
                self.shape = shape
        else:
            self.ncolors = ncolors
            self.shape = shape
            assert shape is not None

        kernel_files = [os.path.join("image", i) for i in self.kernel_files]
        self.compile_kernels(kernel_files,
                             compile_options="-DNB_COLOR=%i" % self.ncolors)
        buffers = [BufferDescription("image1_d", self.shape + (self.ncolors,), numpy.float32, None),
                   BufferDescription("image2_d", self.shape + (self.ncolors,), numpy.float32, None),
                   BufferDescription("max_min_d", 2, numpy.float32, None),
                   ]
        # Temporary buffer for max-min reduction
        self.wg_red = kernel_workgroup_size(self.program, self.kernels.max_min_reduction_stage1)
        if self.wg_red > 1:
            self.wg_red = numpy.int32(1 << int(floor(log(sqrt(numpy.prod(self.shape)), 2))))
            tmp = BufferDescription("tmp_max_min_d", 2 * self.wg_red, numpy.float32, None)
            buffers.append(tmp)
        self.allocate_buffers(buffers, use_array=True)

    def _get_in_out_buffers(self, img, copy=True, out=None):
        """Internal method used to select the proper buffers before processing.

        :param img: expects a numpy array or a pyopencl.array of dim 2 or 3
        :param copy: set to False to directly re-use a pyopencl array
        :param out: provide an output buffer to store the result
        :return: input_buffer, output_buffer
        
        Nota: this is not locked.
        """
        events = []
        if out is not None and isinstance(out, pyopencl.array.Array):
            assert out.shape == self.shape + (self.ncolors,)
            assert out.dtype == numpy.float32
            out.finish()
            output_array = out
        else:
            output_array = self.cl_mem["image2_d"]

        if isinstance(img, pyopencl.array.Array):
            if copy:
                evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["image1_d"].data, img.data)
                input_array = self.cl_mem["image1_d"]
                events.append(EventDescription("copy D->D", evt))
            else:
                img.finish()
                input_array = img
                evt = None
        else:
            # assume this is numpy
            if img.dtype.itemsize > 4:
                logger.warning("Casting to float32 on CPU")
                evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["image1_d"].data, numpy.ascontiguousarray(img, numpy.float32))
                input_array = self.cl_mem["image1_d"]
                events.append(EventDescription("cast+copy H->D", evt))
            else:
                evt = pyopencl.enqueue_copy(self.queue, self.cl_mem["image1_d"].data, numpy.ascontiguousarray(img))
                input_array = self.cl_mem["image1_d"]
                events.append(EventDescription("copy H->D", evt))
        if self.profile:
            self.events += events
        return input_array, output_array

    def to_float(self, img, copy=True, out=None):
        """ Takes any array and convert it to a float array for ease of processing.
        
        :param img: expects a numpy array or a pyopencl.array of dim 2 or 3
        :param copy: set to False to directly re-use a pyopencl array
        :param out: provide an output buffer to store the result
        """
        assert img.shape == self.shape + (self.ncolors,)
        events = []
        with self.sem:
            input_array, output_array = self._get_in_out_buffers(img, copy, out)
            if img.dtype.itemsize > 4:
                # copy device -> device
                ev = pyopencl.enqueue_copy(self.queue, output_array.data, input_array)
                events.append(EventDescription("copy D->D ", ev))
            else:
                # Cast to float:
                name = self.converter[img.dtype]
                kernel = self.kernels.get_kernel(name)
                ev = kernel(self.queue, (self.shape[1], self.shape[0]), None,
                            input_array.data, output_array.data,
                            numpy.int32(self.shape[1]), numpy.int32(self.shape[0])
                            )
                events.append(EventDescription("cast %s" % name, ev))

        if self.profile:
            self.events += events
        if out is None:
            res = output_array.get()
            return res
        else:
            output_array.finish()
            return output_array

    def normalize(self, img, mini=0.0, maxi=1.0, copy=True, out=None):
        """Scale the intensity of the image so that the minimum is 0 and the
        maximum is 1.0 (or any value suggested).
        
        :param img: numpy array or pyopencl array of dim 2 or 3 and of type float
        :param mini: Expected minimum value
        :param maxi: expected maxiumum value
        :param copy: set to False to use directly the input buffer
        :param out: provides an output buffer. prevents a copy D->H
        
        This uses a min/max reduction in two stages plus a map operation  
        """
        assert img.shape == self.shape + (self.ncolors,)
        events = []
        with self.sem:
            input_array, output_array = self._get_in_out_buffers(img, copy, out)
            size = numpy.int32(numpy.prod(self.shape))
            if self.wg_red == 1:
                #  Probably on MacOS CPU WG==1 --> serial code.
                kernel = self.kernels.max_min_serial

                evt = kernel(self.queue, (size,), (1,),
                             input_array.data,
                             size,
                             self.cl_mem["max_min_d"].data)
                events.append(EventDescription("max_min_serial", evt))
            else:
                stage1 = self.kernels.max_min_reduction_stage1
                stage2 = self.kernels.max_min_reduction_stage2
                local_mem = pyopencl.LocalMemory(int(self.wg_red * 8))
                k1 = stage1(self.queue, (int(self.wg_red ** 2),), (int(self.wg_red),),
                            input_array.data,
                            self.cl_mem["tmp_max_min_d"].data,
                            size,
                            local_mem)
                k2 = stage2(self.queue, (int(self.wg_red),), (int(self.wg_red),),
                            self.cl_mem["tmp_max_min_d"].data,
                            self.cl_mem["max_min_d"].data,
                            local_mem)

                events += [EventDescription("max_min_stage1", k1),
                           EventDescription("max_min_stage2", k2)]

            evt = self.kernels.normalize_image(self.queue, (self.shape[1], self.shape[0]), None,
                                               input_array.data, output_array.data,
                                               numpy.int32(self.shape[1]), numpy.int32(self.shape[0]),
                                               self.cl_mem["max_min_d"].data,
                                               numpy.float32(mini), numpy.float32(maxi))
            events.append(EventDescription("normalize", evt))
        if self.profile:
            self.events += events

        if out is None:
            res = output_array.get()
            return res
        else:
            output_array.finish()
            return output_array
