# Type promotion in cuDF

## Numeric data

Our intuition for arithmetic on numeric columns is that the operations
obey the usual algebraic closure rules. Moreover, we usually
implicitly apply the closure rule for integers under division by
extending to rationals.

On a computer, providing a realisation of this intuition is somewhat
challenging, since only floating point types offer algebraic closure
under all operations; integer types are closed under addition,
multiplication, and subtraction as long as no overflow occurs.

In cuDF, we take an approach that broadly matches that of pandas,
however, unlike pandas we generally _do not_ carry out value-dependent
type promotion. That is, we determine a common type for the result of
a binary operation by inspect the types of the operands, but not the
values.

Having said that, this is not quite true. For numerical column binops,
when the `other` value is a scalar, we might look at the value. Under
some slightly complicated rules.

TODO: Should we apply these rules? I think no, or at least, we should
be more consistent about them.

The particularly challenging one is binops that should insert nulls
through division by zero.

What options are available to us here?

Jax implements a nicely principled time promotion scheme that only
ever looks at dtypes, and has the nice property that the promotion
rule is a join semi-lattice on the poset of dtypes. This means,
amongst other things, that type promotion is associative.

Numpy is not quite as simple, though Sebastian Berg is attempting to
tidying things up in NEP50. In general they tend to look at dtypes
only and avoid value-based promotion.

Pandas is very ad-hoc right now, though plausibly there is some
opportunity to help tidy things up. Matthew Roeschke notes that this
is mostly a philosophy of not losing information when users carry out
some operation. As a consequence, pandas is very eager to promote
based on both dtype and value, but does so in a non-associative
(sometimes non-commutative?) way.

### Proposal

Our starting point should be either Jax-like or numpy-like promotion I
think. We can then override on a case-by-case basis if we need to be
closer to pandas. That said, I would like to avoid inspecting values
whereever possible. One issue here is that we can't trap division by
zero in CUDA kernels (as numpy does on the CPU). This is OK for
regular division (since type promotion in that case would result in
floating point output at which point the system is closed and we can
represent infinity and nan), but problematic for `__floordiv__` and
`__mod__` which both stay in integer representations.

Numpy provides runtimewarnings for integer division by zero, and for
overflow (in some cases). Cupy doesn't do so (presumably for the same
reason that cuDF doesn't, namely that the GPU kernel doesn't set an
error flag).

Pandas catches these numpy warnings and then post-processes the result
to promote the float and insert appropriate nan and inf entries.
However, it has so many special cases that it doesn't seem worth
trying to match the behaviour everywhere.

I think what I want is `numpy.result_type(*input_dtypes)` for type
promotion everywhere. With object numeric types treated "weakly" like
Jax. That means that if presented with an untyped literal, we don't
consider its type in the promotion hierarchy (other than integer to
floating point conversions).

This might need occasional handling within cudf to explicitly promote
scalars.

### Decimal columns

How should we handle type promotion here, so far I've just ignored it.

## Non-numeric data

This can be categorised into two types.

1. Leaf column types (string/datetime)
2. Composite columns (struct/list)

Let us assume we have a type promotion rule and permissible binops for
all "leaf" columns. We then need a rule for composite columns.

Status quo:

- struct columns: no binops allowed
- list columns: only `__add__` is supported between two list columns
  with _identical_ types.

Possible implementation choices:

1. Keep the current status quo
2. Extend addition of lists to obey the type promotion rules for
   addition of "leaf" columns. For example, this would promote
   `List(int32) + List(float32)` to `List(float64)` under current
   rules.
3. Support binary operations on composite columns with matching dtype
   "shape", applying promotion rules to the leaves.

I think that (3) is not really a good option, since in many cases it
doesn't really make sense to perform a binary operation between two
composite columns (or at least, it's hard to see a nice principled set
of rules, rather than a large collection of special cases).

My preference is probably for (1), though perhaps (2) is more
ergonomic for users.

I think trying to do anything more complicated is probably not worth
it. Especially since the implementation operates on the leaf columns
anyway, there's no real performance advantage to providing a
high-level API. Operating field-by-field on structs probably also goes
with princple of least surprise (if we can get it right).

If we want to support binops, I think we need to implement `astype`
casting on struct/list columns.
