# Getting started

You will need:

1. Rust development environment. If you use the rapids [combined
   devcontainer](https://github.com/rapidsai/devcontainers/), add
   `"./features/src/rust": {"version": "latest", "profile": "default"},` to your
   preferred configuration. Or else, use
   [rustup](https://www.rust-lang.org/tools/install)
2. A [cudf development
   environment](https://github.com/rapidsai/cudf/blob/branch-24.06/CONTRIBUTING.md#setting-up-your-build-environment).
   The combined devcontainer works, or whatever your favourite approach is.

> ![NOTE] These instructions will get simpler as we merge code in.

## Installing polars

We will need to build polars from source. Until things settle down,
live at `HEAD`.

```sh
git clone https://github.com/pola-rs/polars
cd polars
```

We will install build dependencies in the same environment that we created for
building cudf. Note that polars offers a `make build` command that sets up a
separate virtual environment, but we don't want to do that right now. So in the
polars clone:

```sh
# cudf environment (conda or pip) is active
pip install --upgrade uv
uv pip install --upgrade -r py-polars/requirements-dev.txt
```

Now we have the necessary machinery to build polars
```sh
cd py-polars
# build in debug mode, best option for development/debugging
maturin develop -m Cargo.toml
```

For benchmarking purposes we should build in release mode
```sh
RUSTFLAGS='-C target-cpu=native' maturin develop -m Cargo.toml --release
```

After any update of the polars code, we need to rerun the `maturin` build
command.

## Installing the cudf polars executor

The executor for the polars logical plan lives in the cudf repo, in
`python/cudf_polars`. For now you need [this
branch](https://github.com/wence-/cudf/tree/wence/fea/cudf-polars).

```sh
# get to the cudf repo
cd cudf
git add remote wence https://github.com/wence-/cudf
git fetch wence
git checkout wence/fea/cudf-polars
```

Build cudf as normal and then install the `cudf_polars` package in editable
mode:

```sh
cd python/cudf_polars
pip install --no-deps -e .
```

You should now be able to run the tests in the `cudf_polars` package:
```sh
# -p cudf_polars enables the GPU executor
pytest -p cudf_polars tests/test_basic.py
```

# Executor design

The polars `LogicalPlan` is exposed to python through a low-level
`NodeTraverser` object which offers a callback interface for extracting python
representations of the plan nodes one at a time. The plan intermediate
representation (IR) is composed of plan nodes, which can be thought of as
functions between dataframes: `plan : DataFrame -> DataFrame`. The manipulation
of the dataframes is via expressions which are functions (ignoring
grouping/rolling contexts) from dataframes to columns (or scalars): `expr :
DataFrame -> Column | Scalar`.

## Adding a handler for a new plan node

The plan handler is `_execute_plan` in `cudf_polars/plan.py`, this is a
[`functools.singledispatch`]() function which dispatches on the type of the
first argument. Suppose we want to register a handler for the polars `Cache`
node:

```python
# Module exposing the plan nodes
from polars.polars import nodes

@_execute_plan.register
def _cache(node: nodes.Cache, visitor: PlanVisitor) -> DataFrame:
   # Implementation
```

The `PlanVisitor` object contains some utilities for handling and dealing with
the `NodeTraverser`. To transform a child node we `__call__` the visitor on the
node (which is stored as an opaque integer in the python node):
```python
def _cache(node, visitor):
   # eliding the details of the caching mechanism
   child = visitor(node.input)
   return child
```

As well as child nodes that are plans, most plan nodes contain child
expressions, which should be transformed using the input to the plan as a
context. The `PlanVisitor` contains an `ExprVisitor` which handles the dispatch
for expression nodes. Given an expression, and a dataframe context, we produce a
new column with:
```python
result = visitor.expr_visitor(expression_node, context)
```

As an example, let us look at the implementation of `Filter`:
```python
@_execute_plan.register
def _filter(plan: nodes.Filter, visitor: PlanVisitor):
    result = visitor(plan.input)
    with visitor.record("filter"):
        mask = visitor.expr_visitor(plan.predicate.node, result)
        return result.filter(mask)
```
We first produce the context with `visitor(plan.input)`, we then compute the
mask for the frame with `visitor.expr_visitor(plan.predicate.node, result)`.
Finally, we apply the filter. Any expression reference in a plan node will be a
`polars.polars.expr_nodes.PyExprIR` object. This has two fields, `.output_name`
corresponds to the column name this expression should produce in the output,
`.node` is the node id of the actual expression object.

## Adding a handler for a new expression node

Adding a handle for an expression node is very similar to a plan node. The
evaluator for expressions is implemented in `expressions.py`. `evaluate_expr` is
a `singledispatch` function which should be registered for new values. It has
signature `(Expr, DataFrame, ExprVisitor) -> Column`. Note we cheat a bit
because some expressions can return scalars, so we type pun, and sometimes
things break right now.

The `ExprVisitor` is similar to the `PlanVisitor` and is callable on an
expression node id to transform it. To see how things fit together, let's look
at the implementation of `SortBy` which sorts an expression by some other list
of expressions:
```python
@evaluate_expr.register
def _sort_by(
    expr: expr_nodes.SortBy, context: DataFrame, visitor: ExprVisitor
):
    if visitor.in_groupby:
        raise NotImplementedError("sort_by inside groupby")
    to_sort = visitor(expr.expr, context)
    descending = expr.descending
    sort_keys = [visitor(e, context) for e in expr.by]
    # TODO: no stable to sort_by in polars
    descending, column_order, null_precedence = sort_order(
        descending, nulls_last=True, num_keys=len(sort_keys)
    )
    return plc.sorting.sort_by_key(
        plc.Table([to_sort]),
        plc.Table(sort_keys),
        column_order,
        null_precedence,
    )
```

The visitor object knows whether or not it is inside a groupby context (in which
case the behaviour must be modified), so here we raise an error since that case
is not yet implemented. The column to sort can be obtained by recursing on the
`expr` child. The sort keys are a list obtained by recursing on the `by`
children. Finally, there are some options to the sort and we translate the call
into a pylibcudf primitive. To determine the available slots of the object, we
look at the matching definition in the polars rust code: we don't guarantee API
stability at the moment, so there is little documentation.

## Whole frame expression evaluation

Outside of a group_by or rolling window context, implementing a new handler for
an expression is quite straightforward. We must determine what the
semantics are and map the information contained in an expression node
onto the appropriate (sequence of) pylibcudf calls. Our goal is to
have no dependencies outside of pylibcudf, so try not to use code from
cudf-classic or pyarrow. Some existing code does this, but it is being
removed.
