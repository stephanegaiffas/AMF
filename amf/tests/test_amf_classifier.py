# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: GPL 3.0

# py.test -rA

import numpy as np
import pytest

from amf import AMFClassifier


class TestAMFClassifier(object):

    def test_n_classes(self):
        self.parameter_test_with_min(
            parameter='n_classes',
            valid_val=3,
            invalid_type_val=2.0,
            invalid_val=1,
            min_value=2,
            min_value_str='2',
            mandatory=True,
            fixed_type=int
        )

    def test_n_features(self):
        clf = AMFClassifier(n_classes=2)
        X = np.random.randn(2, 2).astype('float32')
        y = np.array([0., 1.]).astype('float32')
        clf.partial_fit(X, y)
        assert clf.n_features == 2
        with pytest.raises(
                ValueError,
                match="`n_features` is a readonly attribute"
        ):
            clf.n_features = 3

    def test_n_estimators(self):
        self.parameter_test_with_min(
            parameter='n_estimators',
            valid_val=3,
            invalid_type_val=2.0,
            invalid_val=0,
            min_value=1,
            min_value_str='1',
            mandatory=False,
            fixed_type=int
        )

    def test_step(self):
        self.parameter_test_with_min(
            parameter='step',
            valid_val=2.,
            invalid_type_val=0,
            invalid_val=0.,
            min_value_strict=0.,
            min_value_str='0',
            mandatory=False,
            fixed_type=float
        )

    def test_loss(self):
        pass

    def test_use_aggregation(self):
        self.parameter_test_with_type(
            parameter='step',
            valid_val=False,
            invalid_type_val=0,
            mandatory=False,
            fixed_type=bool
        )

    def test_dirichlet(self):
        pass

    def test_split_pure(self):
        self.parameter_test_with_type(
            parameter='split_pure',
            valid_val=False,
            invalid_type_val=0,
            mandatory=False,
            fixed_type=bool
        )

    def test_random_state(self):
        pass

    def test_n_jobs(self):
        self.parameter_test_with_min(
            parameter='n_jobs',
            valid_val=4,
            invalid_type_val=2.0,
            invalid_val=0,
            min_value=1,
            min_value_str='1',
            mandatory=False,
            fixed_type=int
        )

    def test_verbose(self):
        self.parameter_test_with_type(
            parameter='verbose',
            valid_val=False,
            invalid_type_val=0,
            mandatory=False,
            fixed_type=bool
        )

    def test_repr(self):
        amf = AMFClassifier(n_classes=3)
        assert repr(amf) == "AMFClassifier(n_classes=3, n_estimators=10, " \
                            "step=1.0, loss=log, use_aggregation=True, " \
                            "dirichlet=0.01, split_pure=False, n_jobs=1, " \
                            "random_state=0, verbose=True)"

        amf.n_estimators = 42
        assert repr(amf) == "AMFClassifier(n_classes=3, n_estimators=42, " \
                            "step=1.0, loss=log, use_aggregation=True, " \
                            "dirichlet=0.01, split_pure=False, n_jobs=1, " \
                            "random_state=0, verbose=True)"

        amf.verbose = False
        assert repr(amf) == "AMFClassifier(n_classes=3, n_estimators=42, " \
                            "step=1.0, loss=log, use_aggregation=True, " \
                            "dirichlet=0.01, split_pure=False, n_jobs=1, " \
                            "random_state=0, verbose=False)"

    @staticmethod
    def parameter_test_with_min(parameter,
                                valid_val,
                                invalid_type_val,
                                invalid_val,
                                min_value=None,
                                min_value_strict=None,
                                min_value_str=None,
                                mandatory=False,
                                fixed_type=None):
        """Tests for an attribute of integer type

        Parameters
        ----------
        valid_val
            A valid value for the parameter

        invalid_type_val
            A value with invalid type

        invalid_val
            A value which is invalid because of its value

        parameter
        min_value
        mandatory

        Returns
        -------

        """

        def get_params(param, val):
            """If the parameter is not 'n_classes', we need to specify
            `n_classes`, since it's mandatory to create the class
            """
            if param == 'n_classes':
                return {param: val}
            else:
                return {param: val, 'n_classes': 2}

        # If the parameter is mandatory, we check that an exception is raised
        # if not passed to the constructor
        if mandatory:
            with pytest.raises(TypeError) as exc_info:
                AMFClassifier()
            assert exc_info.type is TypeError
            assert exc_info.value.args[0] == "__init__() missing 1 required " \
                                             "positional argument: '%s'" \
                                             % parameter

        if min_value is not None and min_value_strict is not None:
            raise ValueError("You can't set both `min_value` and "
                             "`min_value_strict` at the same time")

        clf = AMFClassifier(**get_params(parameter, valid_val))
        assert getattr(clf, parameter) == valid_val

        # If valid_val is valid, than valid_val + 1 is also valid
        setattr(clf, parameter, valid_val + 1)
        assert getattr(clf, parameter, valid_val + 1)

        with pytest.raises(
                ValueError,
                match="`%s` must be of type `%s`"
                      % (parameter, fixed_type.__name__)
        ):
            setattr(clf, parameter, invalid_type_val)

        with pytest.raises(
                ValueError,
                match="`%s` must be of type `%s`"
                      % (parameter, fixed_type.__name__)
        ):
            AMFClassifier(**get_params(parameter, invalid_type_val))

        if min_value is not None:
            with pytest.raises(
                    ValueError,
                    match="`%s` must be >= %s" % (parameter, min_value_str)
            ):
                setattr(clf, parameter, invalid_val)

            with pytest.raises(
                    ValueError,
                    match="`%s` must be >= %s" % (parameter, min_value_str)
            ):
                AMFClassifier(**get_params(parameter, invalid_val))

        if min_value_strict is not None:
            with pytest.raises(
                    ValueError,
                    match="`%s` must be > %s" % (parameter, min_value_str)
            ):
                setattr(clf, parameter, invalid_val)

            with pytest.raises(
                    ValueError,
                    match="`%s` must be > %s" % (parameter, min_value_str)
            ):
                AMFClassifier(**get_params(parameter, invalid_val))

        clf = AMFClassifier(**get_params(parameter, valid_val))
        # TODO: we should not need to change the dtype here
        X = np.random.randn(2, 2).astype('float32')
        y = np.array([0., 1.]).astype('float32')
        clf.partial_fit(X, y)
        with pytest.raises(ValueError,
                           match="You cannot modify `%s` "
                                 "after calling `partial_fit`" % parameter):
            setattr(clf, parameter, valid_val)

    @staticmethod
    def parameter_test_with_type(parameter,
                                 valid_val,
                                 invalid_type_val,
                                 mandatory,
                                 fixed_type):
        # TODO: code it
        pass
