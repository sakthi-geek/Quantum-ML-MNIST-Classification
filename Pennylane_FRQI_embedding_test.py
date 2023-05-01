import numpy as np
import pennylane as qml
from pennylane.operation import Operation, AnyWires
from pennylane.ops import QubitStateVector
from pennylane.wires import Wires

TOLERANCE = 1e-10


class FRQIEmbedding(Operation):
    num_wires = AnyWires
    grad_method = None

    def __init__(self, features, wires, pad_with=None, normalize=False, do_queue=True, id=None):
        wires = Wires(wires)
        self.pad_with = pad_with
        self.normalize = normalize
        features = self._preprocess(features, wires, pad_with, normalize)
        super().__init__(features, wires=wires, do_queue=do_queue, id=id)

    @property
    def num_params(self):
        return 1

    @property
    def ndim_params(self):
        return (1,)

    @staticmethod
    def compute_decomposition(features, wires):
        return [QubitStateVector(features, wires=wires)]

    @staticmethod
    def _preprocess(features, wires, pad_with, normalize):
        batched = qml.math.ndim(features) > 1

        if batched and qml.math.get_interface(features) == "tensorflow":
            raise ValueError("FRQIEmbedding does not support batched Tensorflow features.")

        features_batch = features if batched else [features]

        new_features_batch = []
        for feature_set in features_batch:
            shape = qml.math.shape(feature_set)

            if len(shape) != 1:
                raise ValueError(f"Features must be a one-dimensional tensor; got shape {shape}.")

            n_features = shape[0]
            dim = 2 ** (len(wires))
            
            if pad_with is None and n_features != dim:
                raise ValueError(
                    f"Features must be of length {dim}; got length {n_features}. "
                    f"Use the 'pad_with' argument for automated padding."
                )

            if pad_with is not None:
                if n_features > dim:
                    raise ValueError(
                        f"Features must be of length {dim} or "
                        f"smaller to be padded; got length {n_features}."
                    )

                if n_features < dim:
                    padding = [pad_with] * (dim - n_features)
                    padding = qml.math.convert_like(padding, feature_set)
                    feature_set = qml.math.hstack([feature_set, padding])

            norm = qml.math.sum(qml.math.abs(feature_set) ** 2)

            if qml.math.is_abstract(norm):
                if normalize or pad_with:
                    feature_set = feature_set / qml.math.sqrt(norm)

            elif not qml.math.allclose(norm, 1.0, atol=TOLERANCE):
                if normalize or pad_with:
                    feature_set = feature_set / qml.math.sqrt(norm)
                elif not normalize:
                    pass
                else:
                    raise ValueError(
                        f"Features must be a vector of norm 1.0; got norm {norm}. "
                        "Use 'normalize=True' to automatically normalize."
                    )

            new_features_batch.append(feature_set)

        return qml.math.cast(
            qml.math.stack(new_features_batch) if batched else new_features_batch[0], np.complex128
        )
