from __future__ import annotations

from pathlib import Path
from typing import Optional

import postbound as pb
from postbound import SqlQuery, JoinTree, PhysicalOperatorAssignment, QueryPlan

from ._util import FgPbConverter
from .context import ContextManager, DatabaseSchema
from .context import CtxGranularity
from .featurization import EncodingInformation, FastgresFeaturization, DatabaseConnection
from .hinting import HintSetFactory, HintLibrary
from .labeling import FastgresLabelProvider
from .model import FastgresContextModel

class FastgresSetting:

    def __init__(
            self,
            pb_connection: pb.postgres.PostgresInterface,
            encoding_information_path: Path,
            training_data_path: Path,
            context_granularity: CtxGranularity,
            hint_library: HintLibrary,
            encoding_workload: pb.Workload,
            feature_skip_columns: Optional[dict[str, list[str]]] = None,  # table, list of columns
    ):
        self.dbc = pb_connection
        self.enc_inf_path = encoding_information_path
        self.label_path = training_data_path
        self.ctx = context_granularity
        self.hint_lib = hint_library
        self.enc_wl = encoding_workload
        self.skip_dict = feature_skip_columns


class FastgresEstimator(pb.PhysicalOperatorSelection, pb.CompleteOptimizationAlgorithm):
    """SafeBound is an upper bound-driven cardinality estimator.

    At its core, SafeBound uses (compressed versions) of most frequent value lists over the join
    columns to calculate the upper bounds. The MCVs are encoded as *piecewise constant functions*
    that are stored in a `SafeBoundCatalog`. The estimator itself acts primarily as a coordinator
    that retrieves the PCFs from the catalog, decomposes the input query, and invokes the FDSB
    algorithm to compute the bound.

    Parameters
    ----------
    catalog: SafeBoundCatalog
        The catalog that stores the piecewise constant functions. The estimator relies on the
        catalog to retrieve the relevant PCFs for a given query.

    See Also
    --------
    SafeBoundCatalog : The catalog that stores the piecewise constant functions
    fdsb : The algorithm that handles the query decomposition and the actual bound calculation

    References
    ----------
    .. Kyle Deeds et al.: "SafeBound: A Practical System for Generating Cardinality Bounds"
      (SIGMOD 2023) https://doi.org/10.1145/3588907
    """

    def optimize_query(self, query: SqlQuery) -> QueryPlan:
        operators = self.select_physical_operators(query)
        hinted_query = self._dbc.hinting().generate_hints(query, physical_operators=operators)
        return self._dbc.optimizer().query_plan(hinted_query)

    def select_physical_operators(self, query: SqlQuery,
                                  join_order: Optional[JoinTree] = None) -> PhysicalOperatorAssignment:
        if self._model is None:
            raise ValueError(
                "Trying to infer physical operators from non existing model. Load or build and fit the model first."
            )
        prediction = self._hsf.hint_set(self._model.predict(query))
        return self._conv.fg_to_pb_hint_set(prediction)

    def __init__(self, setting: FastgresSetting) -> None:
        super().__init__()
        self.setting = setting
        self._dbc = self.setting.dbc
        self._conv = FgPbConverter()
        self._label_provider = FastgresLabelProvider(self.setting.label_path)
        self._ctx_manager = ContextManager(self.setting.ctx, self.setting.enc_wl.queries(),
                                           DatabaseSchema.from_dbc(self._dbc))
        self._enc_skipped = dict() if self.setting.skip_dict is None else self.setting.skip_dict
        self._hsf = HintSetFactory(self.setting.hint_lib)

        self._featurizer = None
        self._model = None

    def load_or_build_featurizer(self, enc_dir: Optional[Path] = None):
        if enc_dir.exists() and any(enc_dir.iterdir()):
            self._featurizer = FastgresFeaturization(self._dbc, self._ctx_manager, enc_dir=enc_dir)
        else:
            enc_info = EncodingInformation.build(DatabaseConnection(self._dbc), self.setting.enc_wl, self._enc_skipped)
            self._featurizer = FastgresFeaturization(self._dbc, self._ctx_manager, enc_info=enc_info)

    def load_or_build_model(self, model_dir: Optional[Path] = None):
        fg_model = FastgresContextModel(
            featurizer=self._featurizer,
            label_provider=self._label_provider,
            context_manager=self._ctx_manager,
            max_depth=1000,
            n_estimators=100,
        )
        if model_dir is not None:
            fg_model.load(model_dir)
        self._model = fg_model

    def fit(self, training_workload: pb.Workload):
        self._model.fit(training_workload)

    def predict(self, test_workload: pb.Workload):
        return self._model.predict_queries(test_workload.queries())

    def save(self, path: Path):
        self._model.save(path)

    def load(self, path: Path):
        self.load_or_build_model(path)

    def describe(self) -> pb.util.jsondict:
        return {
            "name": "Fastgres",
            "version": "1.0",
            "database": self._dbc.describe(),
            "hints": self._hsf.hint_library.hint_list
        }
