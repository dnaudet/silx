# coding: utf-8
# /*##########################################################################
#
# Copyright (c) 2016 European Synchrotron Radiation Facility
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
# ###########################################################################*/
"""Set of icons for buttons.

Use :func:`getQIcon` to create Qt QIcon from the name identifying an icon.
"""

__authors__ = ["T. Vincent"]
__license__ = "MIT"
__date__ = "25/08/2016"


import logging
import weakref
from . import qt
from ..resources import resource_filename


_logger = logging.getLogger(__name__)
"""Module logger"""


_cached_icons = weakref.WeakValueDictionary()
"""Cache loaded icons in a weak structure"""


_supported_formats = None
"""Order of file format extension to check"""


def getQIcon(name):
    """Create a QIcon from its name.

    :param str name: Name of the icon, in one of the defined icons
                     in this module.
    :return: Corresponding QIcon
    :raises: ValueError when name is not known
    """
    if name not in _cached_icons:
        qfile = getQFile(name)
        icon = qt.QIcon(qfile.fileName())
        _cached_icons[name] = icon
    else:
        icon = _cached_icons[name]
    return icon


def getQPixmap(name):
    """Create a QPixmap from its name.

    :param str name: Name of the icon, in one of the defined icons
                     in this module.
    :return: Corresponding QPixmap
    :raises: ValueError when name is not known
    """
    qfile = getQFile(name)
    return qt.QPixmap(qfile.fileName())


def getQFile(name):
    """Create a QFile from an icon name. Filename is found
    according to supported Qt formats.

    :param str name: Name of the icon, in one of the defined icons
                     in this module.
    :return: Corresponding QFile
    :rtype: qt.QFile
    :raises: ValueError when name is not known
    """
    global _supported_formats
    if _supported_formats is None:
        _supported_formats = []
        supported_formats = qt.supportedImageFormats()
        order = ["svg", "png", "jpg"]
        for format in order:
            if format in supported_formats:
                _supported_formats.append(format)
        if len(_supported_formats) == 0:
            _logger.error("No format supported for icons")
        else:
            _logger.debug("Format %s supported", ", ".join(_supported_formats))

    for format in _supported_formats:
        format = str(format)
        filename = resource_filename('gui/icons/%s.%s' % (name, format))
        qfile = qt.QFile(filename)
        if qfile.exists():
            return qfile
    raise ValueError('Not an icon name: %s' % name)
