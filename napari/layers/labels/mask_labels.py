import logging
from itertools import count
from typing import List, Tuple

import numpy as np

from napari.layers.labels.labels import Labels

from ...utils._dtype import normalize_dtype
from ..image._image_utils import guess_multiscale

LOGGER = logging.getLogger("napari.loader")


def _bools_to_int(data) -> int:
    val: int = 0
    for b in data:
        val = (val << 1) | b
    return val


def _int_to_bool_list(data: int, length: int) -> List[bool]:
    arr = [False] * length
    binary_arr = [int(x) for x in list(f'{data:0b}')]
    arr[-len(binary_arr) :] = binary_arr
    return arr


def _collapse_data(data, mask_axis):
    # Use numpy to collapse all data samples down to a single label
    return np.apply_along_axis(_bools_to_int, mask_axis, data)


class MaskLabels(Labels):
    """Labels (or segmentation) layer.

    A set of boolean masks, each defining its own region.
    TODO: Clean up documentation

    Parameters
    ----------
    data : array or list of array
        Labels data as an array or multiscale. Must be bools.
        Please note multiscale rendering is only supported in 2D. In 3D, only
        the lowest resolution scale is displayed.
    num_colors : int
        Number of unique colors to use in colormap.
    features : dict[str, array-like] or DataFrame
        Features table where each row corresponds to a label and each column
        is a feature. The first row corresponds to the background label.
    properties : dict {str: array (N,)} or DataFrame
        Properties for each label. Each property should be an array of length
        N, where N is the number of labels, and the first property corresponds
        to background.
    color : dict of int to str or array
        Custom label to color mapping. Values must be valid color names or RGBA
        arrays.
    seed : float
        Seed for colormap random generator.
    name : str
        Name of the layer.
    metadata : dict
        Layer metadata.
    scale : tuple of float
        Scale factors for the layer.
    translate : tuple of float
        Translation values for the layer.
    rotate : float, 3-tuple of float, or n-D array.
        If a float convert into a 2D rotation matrix using that value as an
        angle. If 3-tuple convert into a 3D rotation matrix, using a yaw,
        pitch, roll convention. Otherwise assume an nD rotation. Angles are
        assumed to be in degrees. They can be converted from radians with
        np.degrees if needed.
    shear : 1-D array or n-D array
        Either a vector of upper triangular values, or an nD shear matrix with
        ones along the main diagonal.
    affine : n-D array or napari.utils.transforms.Affine
        (N+1, N+1) affine transformation matrix in homogeneous coordinates.
        The first (N, N) entries correspond to a linear transform and
        the final column is a length N translation vector and a 1 or a napari
        `Affine` transform object. Applied as an extra transform on top of the
        provided scale, rotate, and shear values.
    opacity : float
        Opacity of the layer visual, between 0.0 and 1.0.
    blending : str
        One of a list of preset blending modes that determines how RGB and
        alpha values of the layer visual get mixed. Allowed values are
        {'opaque', 'translucent', and 'additive'}.
    rendering : str
        3D Rendering mode used by vispy. Must be one {'translucent', 'iso_categorical'}.
        'translucent' renders without lighting. 'iso_categorical' uses isosurface
        rendering to calculate lighting effects on labeled surfaces.
        The default value is 'iso_categorical'.
    visible : bool
        Whether the layer visual is currently being displayed.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. If not specified by
        the user and if the data is a list of arrays that decrease in shape
        then it will be taken to be multiscale. The first image in the list
        should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    cache : bool
        Whether slices of out-of-core datasets should be cached upon retrieval.
        Currently, this only applies to dask arrays.
    experimental_slicing_plane : dict or SlicingPlane
        Properties defining plane rendering in 3D. Properties are defined in
        data coordinates. Valid dictionary keys are
        {'position', 'normal', 'thickness', and 'enabled'}.
    experimental_clipping_planes : list of dicts, list of ClippingPlane, or ClippingPlaneList
        Each dict defines a clipping plane in 3D in data coordinates.
        Valid dictionary keys are {'position', 'normal', and 'enabled'}.
        Values on the negative side of the normal are discarded if the plane is enabled.

    Attributes
    ----------
    data : array or list of array
        Integer label data as an array or multiscale. Can be N dimensional.
        Every pixel contains an integer ID corresponding to the region it
        belongs to. The label 0 is rendered as transparent. Please note
        multiscale rendering is only supported in 2D. In 3D, only
        the lowest resolution scale is displayed.
    multiscale : bool
        Whether the data is a multiscale image or not. Multiscale data is
        represented by a list of array like image data. The first image in the
        list should be the largest. Please note multiscale rendering is only
        supported in 2D. In 3D, only the lowest resolution scale is
        displayed.
    metadata : dict
        Labels metadata.
    num_colors : int
        Number of unique colors to use in colormap.
    features : Dataframe-like
        Features table where each row corresponds to a label and each column
        is a feature. The first row corresponds to the background label.
    properties : dict {str: array (N,)}, DataFrame
        Properties for each label. Each property should be an array of length
        N, where N is the number of labels, and the first property corresponds
        to background.
    color : dict of int to str or array
        Custom label to color mapping. Values must be valid color names or RGBA
        arrays.
    seed : float
        Seed for colormap random generator.
    opacity : float
        Opacity of the labels, must be between 0 and 1.
    contiguous : bool
        If `True`, the fill bucket changes only connected pixels of same label.
    n_edit_dimensions : int
        The number of dimensions across which labels will be edited.
    contour : int
        If greater than 0, displays contours of labels instead of shaded regions
        with a thickness equal to its value.
    brush_size : float
        Size of the paint brush in data coordinates.
    selected_label : int
        Index of selected label. Can be greater than the current maximum label.
    mode : str
        Interactive mode. The normal, default mode is PAN_ZOOM, which
        allows for normal interactivity with the canvas.

        In PICK mode the cursor functions like a color picker, setting the
        clicked on label to be the current label. If the background is picked it
        will select the background label `0`.

        In PAINT mode the cursor functions like a paint brush changing any
        pixels it brushes over to the current label. If the background label
        `0` is selected than any pixels will be changed to background and this
        tool functions like an eraser. The size and shape of the cursor can be
        adjusted in the properties widget.

        In FILL mode the cursor functions like a fill bucket replacing pixels
        of the label clicked on with the current label. It can either replace
        all pixels of that label or just those that are contiguous with the
        clicked on pixel. If the background label `0` is selected than any
        pixels will be changed to background and this tool functions like an
        eraser.

        In ERASE mode the cursor functions similarly to PAINT mode, but to
        paint with background label, which effectively removes the label.
    experimental_slicing_plane : SlicingPlane
        Properties defining plane rendering in 3D.
    experimental_clipping_planes : ClippingPlaneList
        Clipping planes defined in data coordinates, used to clip the volume.

    Notes
    -----
    _selected_color : 4-tuple or None
        RGBA tuple of the color of the selected label, or None if the
        background label `0` is selected.
    """

    _history_limit = 100

    def __init__(
        self,
        data: np.ndarray,
        mask_axis: int = -1,
        *,
        num_colors=50,
        features=None,
        properties=None,
        color=None,
        seed=0.5,
        name=None,
        metadata=None,
        scale=None,
        translate=None,
        rotate=None,
        shear=None,
        affine=None,
        opacity=0.7,
        blending='translucent',
        rendering='iso_categorical',
        visible=True,
        multiscale=None,
        cache=True,
        experimental_slicing_plane=None,
        experimental_clipping_planes=None,
    ):

        self._mask_axis = mask_axis
        self._label_map: dict[Tuple, int] = {}
        self._label_gen = count(start=0)
        self.masks = self._ensure_bool_labels(data)
        self._collapsed = _collapse_data(
            self.masks,
            self._mask_axis,
        )

        super().__init__(
            self._collapsed,
            num_colors=num_colors,
            features=features,
            properties=properties,
            color=color,
            seed=seed,
            name=name,
            metadata=metadata,
            scale=scale,
            translate=translate,
            rotate=rotate,
            shear=shear,
            affine=affine,
            opacity=opacity,
            blending=blending,
            rendering=rendering,
            visible=visible,
            multiscale=multiscale,
            cache=cache,
            experimental_slicing_plane=experimental_slicing_plane,
            experimental_clipping_planes=experimental_clipping_planes,
        )

    # TODO: Can we reconcile this with the _ensure_int_labels in labels.py?
    def _ensure_bool_labels(self, data):
        """Ensure data is integer by converting from bool if required, raising an error otherwise."""
        looks_multiscale, data = guess_multiscale(data)
        if not looks_multiscale:
            data = [data]
        bool_data = []
        for data_level in data:
            # normalize_dtype turns e.g. tensorstore or torch dtypes into
            # numpy dtypes
            dtype = normalize_dtype(data_level.dtype)
            if np.issubdtype(dtype, np.bool_):
                bool_data.append(data_level)
            else:
                LOGGER.warning(
                    "MaskLabels must be of boolean type: passed dtype %s will be converted into booleans!",
                    str(dtype),
                )
                bool_data.append(data_level.astype(bool))
        data = bool_data
        if not looks_multiscale:
            data = data[0]
        return data
