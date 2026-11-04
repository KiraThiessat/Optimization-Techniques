
""" Fastgres: Making learned query optimizer hinting effective

Fastgres is a learned enumerator that uses query optimizers hints to steer the native optimizer to choose different
plans than it normally would. To do so, Fastgres pregenerates query labels that represent the best possible hint set
in its search space. Afterward, these labels are learned by a contextual gradient boosting classifier, which is a
gradient boosting model for each query context that can be defined in various granularity.

We implement Fastgres from the latest available


High-level API
---------------

SafeBound implements the PostBOUND
`CardinalityEstimator <https://postbound.readthedocs.io/en/latest/api/core/optimization-pipelines.html#postbound.CardinalityEstimator>`__
interface. The `SafeBoundEstimator` acts as the central entrypoint. It requires a
`SafeBoundCatalog` to function correctly. This catalog stores all individual (conditioned and
unconditioned) piecewise constant functions. A catalog can be created either based on a workload,
or inferred completely from the database. See its documentation for more details.



Pregenerating Fastgres Labels and Encoders
--------------------------


Example
-------

.. code:: python

    #
    # Step 1: generic setup
    #
    # As usual for PostBOUND, connect to our target database and load the workload
    import postbound as pb
    import postbound_extensions as pbx

    pg_instance = pb.postgres.connect(config_file=".pg-connect-stats.toml")
    stats = pb.workloads.stats()


    #
    # Step 2: build our catalog
    #
    # We use the same "hyperparameters" (mcv size, histogram depth, etc.)
    # as the original papers
    safebound_spec = pbx.safebound.SafeBoundSpec.default()

    # If no catalog exists, build a new one fine-tuned for the workload
    # otherwise, reload the existing catalog
    cat = pbx.safebound.SafeBoundCatalog.load_or_build(
        "models/safebound/stats", database=pg_instance, workload=stats, spec=safebound_spec
    )


    #
    # Step 3: create the actual estimator
    #
    safebound = pbx.safebound.SafeBoundEstimator(cat)


    #
    # Step 4: Evaluation
    #
    # We can already use this optimizer to calculate the bound of different queries:
    query = stats["q-10"]
    print(safebound.calculate_estimate(query, query.tables()))

    # Or we can run the PostBOUND benchmarking utilities
    pb.bench.execute_workload(
        stats,
        on=safebound,
        query_preparation={"analyze": True, "prewarm": True},
        name="safebound-test",
        progressive_output="safebound-benchmark.parquet",
        logger="tqdm"
    )


Deviations from the Original Paper
----------------------------------


References
----------

.. [Deeds2023] Kyle Deeds et al.: "SafeBound: A Practical System for Generating Cardinality Bounds"
   SIGMOD 2023, https://doi.org/10.1145/3588907
.. Reference Implementation: https://github.com/kylebd99/SafeBound
"""


from ._fastgres import FastgresEstimator, FastgresSetting
from ._util import FgPbConverter
from .context import (
    ColumnContext,
    ContextManager,
    CtxGranularity,
    DatabaseSchema,
    SchemaContext,
    SuperTableContext,
    TableContext,
)
from .featurization import (
    DatabaseConnection,
    EncodingInformation,
    FastgresFeaturization,
)
from .hinting import CORE_HINT_LIBRARY, HintSet, HintSetFactory
from .labeling import (
    FastgresLabelProvider,
    FastLabelSettings,
    QueryLabeling,
    WorkloadLabeling,
    WorkloadLabelSettings,
)
from .model import FastgresContextModel, FastgresModel

__all__ = [
    "CORE_HINT_LIBRARY",
    "ColumnContext",
    "ContextManager",
    "CtxGranularity",
    "DatabaseConnection",
    "DatabaseSchema",
    "EncodingInformation",
    "FastLabelSettings",
    "FastgresContextModel",
    "FastgresEstimator",
    "FastgresFeaturization",
    "FastgresLabelProvider",
    "FastgresModel",
    "FastgresSetting",
    "FgPbConverter",
    "HintSet",
    "HintSetFactory",
    "QueryLabeling",
    "SchemaContext",
    "SuperTableContext",
    "TableContext",
    "WorkloadLabelSettings",
    "WorkloadLabeling",
]
