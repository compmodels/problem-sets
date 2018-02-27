from __future__ import division

import matplotlib
import matplotlib.colors
import matplotlib.markers
import numpy as np
import six
import warnings


try:
    _named_colors = matplotlib.colors.ColorConverter.colors.copy()
    for colorname, hexcode in matplotlib.colors.cnames.items():
        _named_colors[colorname] = matplotlib.colors.hex2color(hexcode)
except: # pragma: no cover
    warnings.warn("Could not get matplotlib colors, named colors will not be available")
    _named_colors = {}


class InvalidPlotError(Exception):
    pass


class PlotChecker(object):
    """A generic object to test plots.
    Parameters
    ----------
    axis : ``matplotlib.axes.Axes`` object
        A set of matplotlib axes (e.g. obtained through ``plt.gca()``)
    """

    _named_colors = _named_colors

    def __init__(self, axis):
        """Initialize the PlotChecker object."""
        self.axis = axis

    @classmethod
    def _color2rgb(cls, color):
        """Converts the given color to a 3-tuple RGB color.
        Parameters
        ----------
        color :
            Either a matplotlib color name (e.g. ``'r'`` or ``'red'``), a
            hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or a 4-tuple RGBA
            color.
        Returns
        -------
        rgb : 3-tuple RGB color
        """
        if isinstance(color, six.string_types):
            if color in cls._named_colors:
                return tuple(cls._named_colors[color])
            else:
                return tuple(matplotlib.colors.hex2color(color))
        elif hasattr(color, '__iter__') and len(color) == 3:
            return tuple(float(x) for x in color)
        elif hasattr(color, '__iter__') and len(color) == 4:
            return tuple(float(x) for x in color[:3])
        else:
            raise ValueError("Invalid color: {}".format(color))

    @classmethod
    def _color2alpha(cls, color):
        """Converts the given color to an alpha value. For all cases except
        RGBA colors, this value will be 1.0.
        Parameters
        ----------
        color :
            Either a matplotlib color name (e.g. ``'r'`` or ``'red'``), a
            hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or a 4-tuple RGBA
            color.
        Returns
        -------
        alpha : float
        """
        if isinstance(color, six.string_types):
            return 1.0
        elif hasattr(color, '__iter__') and len(color) == 3:
            return 1.0
        elif hasattr(color, '__iter__') and len(color) == 4:
            return float(color[3])
        else:
            raise ValueError("Invalid color: {}".format(color))

    @classmethod
    def _parse_marker(cls, marker):
        """Converts the given marker to a consistent marker type. In practice,
        this is basically just making sure all null markers (``''``, ``'None'``,
        ``None``) get converted to empty strings.
        Parameters
        ----------
        marker : string
            The marker type
        Returns
        -------
        marker : string
        """
        if marker is None or marker == 'None':
            return ''
        return marker

    @classmethod
    def _tile_or_trim(cls, x, y):
        """Tiles or trims the first dimension of ``y`` so that ``x.shape[0]`` ==
        ``y.shape[0]``.
        Parameters
        ----------
        x : array-like
            A numpy array with any number of dimensions.
        y : array-like
            A numpy array with any number of dimensions.
        """
        xn = x.shape[0]
        yn = y.shape[0]
        if xn > yn:
            numrep = int(np.ceil(xn / yn))
            y = np.tile(y, (numrep,) + (1,) * (y.ndim - 1))
            yn = y.shape[0]
        if xn < yn:
            y = y[:xn]
        return y

    @property
    def title(self):
        """The title of the matplotlib plot, stripped of whitespace."""
        return self.axis.get_title().strip()

    def assert_title_equal(self, title):
        """Asserts that the given title is the same as the plotted
        :attr:`~plotchecker.PlotChecker.title`.
        Parameters
        ----------
        title : string
            The expected title
        """
        title = title.strip()
        if self.title != title:
            raise AssertionError(
                "title is incorrect: '{}'' (expected '{}')".format(
                    self.title, title))

    def assert_title_exists(self):
        """Asserts that the plotted :attr:`~plotchecker.PlotChecker.title` is
        non-empty.
        """
        if self.title == '':
            raise AssertionError("no title")

    @property
    def xlabel(self):
        """The xlabel of the matplotlib plot, stripped of whitespace."""
        return self.axis.get_xlabel().strip()

    def assert_xlabel_equal(self, xlabel):
        """Asserts that the given xlabel is the same as the plotted
        :attr:`~plotchecker.PlotChecker.xlabel`.
        Parameters
        ----------
        xlabel : string
            The expected xlabel
        """
        xlabel = xlabel.strip()
        if self.xlabel != xlabel:
            raise AssertionError(
                "xlabel is incorrect: '{}'' (expected '{}')".format(
                    self.xlabel, xlabel))

    def assert_xlabel_exists(self):
        """Asserts that the plotted :attr:`~plotchecker.PlotChecker.xlabel` is
        non-empty.
        """
        if self.xlabel == '':
            raise AssertionError("no xlabel")

    @property
    def ylabel(self):
        """The ylabel of the matplotlib plot, stripped of whitespace."""
        return self.axis.get_ylabel().strip()

    def assert_ylabel_equal(self, ylabel):
        """Asserts that the given ylabel is the same as the plotted
        :attr:`~plotchecker.PlotChecker.ylabel`.
        Parameters
        ----------
        ylabel : string
            The expected ylabel
        """
        ylabel = ylabel.strip()
        if self.ylabel != ylabel:
            raise AssertionError(
                "ylabel is incorrect: '{}'' (expected '{}')".format(
                    self.ylabel, ylabel))

    def assert_ylabel_exists(self):
        """Asserts that the plotted :attr:`~plotchecker.PlotChecker.ylabel` is
        non-empty.
        """
        if self.ylabel == '':
            raise AssertionError("no ylabel")

    @property
    def xlim(self):
        """The x-axis limits of the matplotlib plot."""
        return self.axis.get_xlim()

    def assert_xlim_equal(self, xlim):
        """Asserts that the given xlim is the same as the plot's
        :attr:`~plotchecker.PlotChecker.xlim`.
        Parameters
        ----------
        xlim : 2-tuple
            The expected xlim
        """
        if self.xlim != xlim:
            raise AssertionError(
                "xlim is incorrect: {} (expected {})".format(
                    self.xlim, xlim))

    @property
    def ylim(self):
        """The y-axis limits of the matplotlib plot."""
        return self.axis.get_ylim()

    def assert_ylim_equal(self, ylim):
        """Asserts that the given ylim is the same as the plot's
        :attr:`~plotchecker.PlotChecker.ylim`.
        Parameters
        ----------
        ylim : 2-tuple
            The expected ylim
        """
        if self.ylim != ylim:
            raise AssertionError(
                "ylim is incorrect: {} (expected {})".format(
                    self.ylim, ylim))

    @property
    def xticks(self):
        """The tick locations along the plot's x-axis."""
        return self.axis.get_xticks()

    def assert_xticks_equal(self, xticks):
        """Asserts that the given xticks are the same as the plot's
        :attr:`~plotchecker.PlotChecker.xticks`.
        Parameters
        ----------
        xticks : list
            The expected tick locations on the x-axis
        """
        np.testing.assert_equal(self.xticks, xticks)

    @property
    def yticks(self):
        """The tick locations along the plot's y-axis."""
        return self.axis.get_yticks()

    def assert_yticks_equal(self, yticks):
        """Asserts that the given yticks are the same as the plot's
        :attr:`~plotchecker.PlotChecker.yticks`.
        Parameters
        ----------
        yticks : list
            The expected tick locations on the y-axis
        """
        np.testing.assert_equal(self.yticks, yticks)

    @property
    def xticklabels(self):
        """The tick labels along the plot's x-axis, stripped of whitespace."""
        return [x.get_text().strip() for x in self.axis.get_xticklabels()]

    def assert_xticklabels_equal(self, xticklabels):
        """Asserts that the given xticklabels are the same as the plot's
        :attr:`~plotchecker.PlotChecker.xticklabels`.
        Parameters
        ----------
        xticklabels : list
            The expected tick labels on the x-axis
        """
        xticklabels = [x.strip() for x in xticklabels]
        np.testing.assert_equal(self.xticklabels, xticklabels)

    @property
    def yticklabels(self):
        """The tick labels along the plot's y-axis, stripped of whitespace."""
        return [x.get_text().strip() for x in self.axis.get_yticklabels()]

    def assert_yticklabels_equal(self, yticklabels):
        """Asserts that the given yticklabels are the same as the plot's
        :attr:`~plotchecker.PlotChecker.yticklabels`.
        Parameters
        ----------
        yticklabels : list
            The expected tick labels on the y-axis
        """
        yticklabels = [y.strip() for y in yticklabels]
        np.testing.assert_equal(self.yticklabels, yticklabels)


    @property
    def _texts(self):
        """All ``matplotlib.text.Text`` objects in the plot, excluding titles."""
        texts = []
        for x in self.axis.get_children():
            if not isinstance(x, matplotlib.text.Text):
                continue
            if x == self.axis.title:
                continue
            if x == getattr(self.axis, '_left_title', None):
                continue
            if x == getattr(self.axis, '_right_title', None):
                continue
            texts.append(x)
        return texts

    @property
    def textlabels(self):
        """The labels of all ``matplotlib.text.Text`` objects in the plot, excluding titles."""
        return [x.get_text().strip() for x in self._texts]

    def assert_textlabels_equal(self, textlabels):
        """Asserts that the given textlabels are the same as the plot's
        :attr:`~plotchecker.PlotChecker.textlabels`.
        Parameters
        ----------
        textlabels : list
            The expected text labels on the plot
        """
        textlabels = [x.strip() for x in textlabels]
        np.testing.assert_equal(self.textlabels, textlabels)

    @property
    def textpoints(self):
        """The locations of all ``matplotlib.text.Text`` objects in the plot, excluding titles."""
        return np.vstack([x.get_position() for x in self._texts])

    def assert_textpoints_equal(self, textpoints):
        """Asserts that the given locations of the text objects are the same as
        the plot's :attr:`~plotchecker.PlotChecker.textpoints`.
        Parameters
        ----------
        textpoints : array-like, N-by-2
            The expected text locations on the plot, where the first column
            corresponds to the x-values, and the second column corresponds to
            the y-values.
        """
        np.testing.assert_equal(self.textpoints, textpoints)

    def assert_textpoints_allclose(self, textpoints, **kwargs):
        """Asserts that the given locations of the text objects are almost the
        same as the plot's :attr:`~plotchecker.PlotChecker.textpoints`.
        Parameters
        ----------
        textpoints : array-like, N-by-2
            The expected text locations on the plot, where the first column
            corresponds to the x-values, and the second column corresponds to
            the y-values.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``
        """
        np.testing.assert_allclose(self.textpoints, textpoints, **kwargs)

class ScatterPlotChecker(PlotChecker):
    """A plot checker for scatter plots.
    Parameters
    ----------
    axis : ``matplotlib.axes.Axes`` object
        A set of matplotlib axes (e.g. obtained through ``plt.gca()``)
    """

    def __init__(self, axis):
        """Initialize the scatter plot checker."""

        super(ScatterPlotChecker, self).__init__(axis)
        self.lines = self.axis.get_lines()
        self.collections = self.axis.collections

        # check that there are only lines or collections, not both
        if len(self.lines) == 0 and len(self.collections) == 0:
            raise InvalidPlotError("No data found")

        # check that if there are lines, linestyle is '' and markers are not ''
        for x in self.lines:
            if len(x.get_xydata()) > 1 and x.get_linestyle() != 'None':
                raise InvalidPlotError("This is supposed to be a scatter plot, but it has lines!")
            if self._parse_marker(x.get_marker()) == '':
                raise InvalidPlotError("This is supposed to be a scatter plot, but there are no markers!")

    def _parse_expected_attr(self, attr_name, attr_val):
        """Ensure that the given expected attribute values are in the right shape."""
        if attr_name in ('colors', 'edgecolors'):
            # if it's a color, first check if it's just a single color -- if it's
            # not a single color, this command will throw an error and we can try
            # iterating over the multiple colors that were given
            try:
                attr_val = np.array([self._color2rgb(attr_val)])
            except (ValueError, TypeError):
                attr_val = np.array([self._color2rgb(x) for x in attr_val])

        elif not hasattr(attr_val, '__iter__'):
            # if it's not a color, then just make sure we have an array
            attr_val = np.array([attr_val])

        # tile the given values if we've only been given one, so it's the same
        # shape as the data
        if len(attr_val) == 1:
            attr_val = self._tile_or_trim(self.x_data, attr_val)

        return attr_val

    def assert_num_points(self, num_points):
        """Assert that the plot has the given number of points.
        Parameters
        ----------
        num_points : int
        """
        if num_points != len(self.x_data):
            raise AssertionError(
                "Plot has incorrect number of points: {} (expected {})".format(
                    len(self.x_data), num_points))

    @property
    def x_data(self):
        """The x-values of the plotted data (1-D array)."""
        all_x_data = []
        if len(self.lines) > 0:
            all_x_data.append(np.concatenate([x.get_xydata()[:, 0] for x in self.lines]))
        if len(self.collections) > 0:
            all_x_data.append(np.concatenate([x.get_offsets()[:, 0] for x in self.collections]))
        return np.concatenate(all_x_data, axis=0)

    def assert_x_data_equal(self, x_data):
        """Assert that the given x-data is equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.x_data`.
        Parameters
        ----------
        x_data : 1-D array-like
            The expected x-data. The number of elements should be equal to the
            (expected) number of plotted points.
        """
        np.testing.assert_equal(self.x_data, x_data)

    def assert_x_data_allclose(self, x_data, **kwargs):
        """Assert that the given x-data is almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.x_data`.
        Parameters
        ----------
        x_data : 1-D array-like
            The expected x-data. The number of elements should be equal to the
            (expected) number of plotted points.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``
        """
        np.testing.assert_allclose(self.x_data, x_data, **kwargs)

    @property
    def y_data(self):
        """The y-values of the plotted data (1-D array)."""
        all_y_data = []
        if len(self.lines) > 0:
            all_y_data.append(np.concatenate([x.get_xydata()[:, 1] for x in self.lines]))
        if len(self.collections) > 0:
            all_y_data.append(np.concatenate([x.get_offsets()[:, 1] for x in self.collections]))
        return np.concatenate(all_y_data, axis=0)

    def assert_y_data_equal(self, y_data):
        """Assert that the given y-data is equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.y_data`.
        Parameters
        ----------
        y_data : 1-D array-like
            The expected y-data. The number of elements should be equal to the
            (expected) number of plotted points.
        """
        np.testing.assert_equal(self.y_data, y_data)

    def assert_y_data_allclose(self, y_data, **kwargs):
        """Assert that the given y-data is almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.y_data`.
        Parameters
        ----------
        y_data : 1-D array-like
            The expected y-data. The number of elements should be equal to the
            (expected) number of plotted points.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``
        """
        np.testing.assert_allclose(self.y_data, y_data, **kwargs)

    @property
    def colors(self):
        """The colors of the plotted points. Columns correspond to RGB values."""
        all_colors = []

        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                colors = np.array([self._color2rgb(x.get_markerfacecolor())])
                all_colors.append(self._tile_or_trim(points, colors))

        if len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                colors = np.array([self._color2rgb(i) for i in x.get_facecolors()])
                all_colors.append(self._tile_or_trim(points, colors))

        return np.concatenate(all_colors, axis=0)

    def assert_colors_equal(self, colors):
        """Assert that the given colors are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.colors`.
        Parameters
        ----------
        colors : single color, or list of expected line colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.
        """
        np.testing.assert_equal(
            self.colors,
            self._parse_expected_attr("colors", colors))

    def assert_colors_allclose(self, colors, **kwargs):
        """Assert that the given colors are almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.colors`.
        Parameters
        ----------
        colors : single color, or list of expected line colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``
        """
        np.testing.assert_allclose(
            self.colors,
            self._parse_expected_attr("colors", colors),
            **kwargs)

    @property
    def alphas(self):
        """The alpha values of the plotted points."""
        all_alphas = []

        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                if x.get_alpha() is None:
                    alpha = np.array([self._color2alpha(x.get_markerfacecolor())])
                else:
                    alpha = np.array([x.get_alpha()])
                all_alphas.append(self._tile_or_trim(points, alpha))

        if len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                if x.get_alpha() is None:
                    alpha = np.array([self._color2alpha(i) for i in x.get_facecolors()])
                else:
                    alpha = np.array([x.get_alpha()])
                all_alphas.append(self._tile_or_trim(points, alpha))

        return np.concatenate(all_alphas)

    def assert_alphas_equal(self, alphas):
        """Assert that the given alpha values are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.alphas`.
        Parameters
        ----------
        alphas :
            The expected alpha values. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.
        """
        np.testing.assert_equal(
            self.alphas, self._parse_expected_attr("alphas", alphas))

    def assert_alphas_allclose(self, alphas, **kwargs):
        """Assert that the given alpha values are almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.alphas`.
        Parameters
        ----------
        alphas :
            The expected alpha values. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``
        """
        np.testing.assert_allclose(
            self.alphas,
            self._parse_expected_attr("alphas", alphas),
            **kwargs)

    @property
    def edgecolors(self):
        """The edge colors of the plotted points. Columns correspond to RGB values."""
        all_colors = []

        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                colors = np.array([self._color2rgb(x.get_markeredgecolor())])
                all_colors.append(self._tile_or_trim(points, colors))

        if len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                colors = np.array([self._color2rgb(i) for i in x.get_edgecolors()])
                all_colors.append(self._tile_or_trim(points, colors))

        return np.concatenate(all_colors, axis=0)

    def assert_edgecolors_equal(self, edgecolors):
        """Assert that the given edge colors are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.edgecolors`.
        Parameters
        ----------
        edgecolors : single color, or list of expected edge colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.
        """
        np.testing.assert_equal(
            self.edgecolors,
            self._parse_expected_attr("edgecolors", edgecolors))

    def assert_edgecolors_allclose(self, edgecolors, **kwargs):
        """Assert that the given edge colors are almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.edgecolors`.
        Parameters
        ----------
        edgecolors : single color, or list of expected edge colors
            Each color can be either a matplotlib color name (e.g. ``'r'`` or
            ``'red'``), a hexcode (e.g. ``"#FF0000"``), a 3-tuple RGB color, or
            a 4-tuple RGBA color.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``
        """
        np.testing.assert_allclose(
            self.edgecolors,
            self._parse_expected_attr("edgecolors", edgecolors),
            **kwargs)

    @property
    def edgewidths(self):
        """The edge widths of the plotted points."""
        all_colors = []

        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                colors = np.array([x.get_markeredgewidth()])
                all_colors.append(self._tile_or_trim(points, colors))

        if len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                colors = np.array(x.get_linewidths())
                all_colors.append(self._tile_or_trim(points, colors))

        return np.concatenate(all_colors, axis=0)

    def assert_edgewidths_equal(self, edgewidths):
        """Assert that the given edge widths are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.edgewidths`.
        Parameters
        ----------
        edgewidths :
            The expected edge widths. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.
        """
        np.testing.assert_equal(
            self.edgewidths,
            self._parse_expected_attr("edgewidths", edgewidths))

    def assert_edgewidths_allclose(self, edgewidths, **kwargs):
        """Assert that the given edge widths are almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.edgewidths`.
        Parameters
        ----------
        edgewidths :
            The expected edge widths. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``
        """
        np.testing.assert_allclose(
            self.edgewidths,
            self._parse_expected_attr("edgewidths", edgewidths),
            **kwargs)

    @property
    def sizes(self):
        """The size of the plotted points. This is the square of
        :attr:`~plotchecker.ScatterPlotChecker.markersizes`.
        """
        all_sizes = []

        if len(self.lines) > 0:
            for x in self.lines:
                points = x.get_xydata()
                sizes = np.array([x.get_markersize() ** 2])
                all_sizes.append(self._tile_or_trim(points, sizes))

        if len(self.collections) > 0:
            for x in self.collections:
                points = x.get_offsets()
                sizes = x.get_sizes()
                all_sizes.append(self._tile_or_trim(points, sizes))

        return np.concatenate(all_sizes, axis=0)

    def assert_sizes_equal(self, sizes):
        """Assert that the given point sizes are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.sizes`.
        Parameters
        ----------
        sizes :
            The expected point sizes. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.
        """
        np.testing.assert_equal(
            self.sizes,
            self._parse_expected_attr("sizes", sizes))

    def assert_sizes_allclose(self, sizes, **kwargs):
        """Assert that the given point sizes are almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.sizes`.
        Parameters
        ----------
        sizes :
            The expected point sizes. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``
        """
        np.testing.assert_allclose(
            self.sizes,
            self._parse_expected_attr("sizes", sizes),
            **kwargs)

    @property
    def markersizes(self):
        """The marker size of the plotted points. This is the square root of
        :attr:`~plotchecker.ScatterPlotChecker.sizes`.
        """
        return np.sqrt(self.sizes)

    def assert_markersizes_equal(self, markersizes):
        """Assert that the given marker sizes are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.markersizes`.
        Parameters
        ----------
        markersizes :
            The expected marker sizes. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.
        """
        np.testing.assert_equal(
            self.markersizes,
            self._parse_expected_attr("markersizes", markersizes))

    def assert_markersizes_allclose(self, markersizes, **kwargs):
        """Assert that the given marker sizes are almost equal to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.markersizes`.
        Parameters
        ----------
        markersizes :
            The expected marker sizes. This should either be a single number
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.
        kwargs :
            Additional keyword arguments to pass to
            ``numpy.testing.assert_allclose``
        """
        np.testing.assert_allclose(
            self.markersizes,
            self._parse_expected_attr("markersizes", markersizes),
            **kwargs)

    @property
    def markers(self):
        """The marker styles of the plotted points. Unfortunately, this
        information is currently unrecoverable from matplotlib, and so this
        attribute is not actually implemented.
        """
        raise NotImplementedError("markers are unrecoverable for scatter plots")

    def assert_markers_equal(self, markers):
        """Assert that the given marker styles are equivalent to the plotted
        :attr:`~plotchecker.ScatterPlotChecker.markers`.
        Note: information about marker style is currently unrecoverable from
        collections in matplotlib, so this method is not actually implemented.
        Parameters
        ----------
        markers :
            The expected marker styles. This should either be a single style
            (which will apply to all the points) or an array with size equal to
            the number of (expected) points.
        """
        np.testing.assert_equal(
self.markers, self._parse_expected_attr("markers", markers))