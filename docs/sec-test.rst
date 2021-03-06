Steinbruch
==========

Ok, here is a formula with a label

.. math::
    f(x) = W x + b
    :label: test

.. math::
    \begin{align}
        F(b) - F(a) &= \int_a^b f(x) dx
    \end{align}
    :label: test1

And we can refer to this equation as Eq. :eq:`test`. And this is an inline
formula :math:`f(x)`. It is also possible to refer to :eq:`test1`.

.. todo::
    We really need to repair the eqautions references.

We can immediately plot the function. Here, for :math:`W=2` and :math:`b=1`:

.. plot:: sec-test/matplot.py
    :include-source:

For the plot we have used the code:

.. code-block:: python
    :linenos:

    def sum(a, b):
        return a + b

Or files can be sourced:

.. literalinclude:: sec-test/matplot.py
    :linenos:

This is something taken from [TESTCIT2016]_

Here is some JavaScript

.. raw:: html
    :file: sec-test/buttons.html

And here an interactive plot:

.. raw:: html
    :file: sec-test/plot.html

.. _sec-thisone:

This is a section
-----------------

How can we get numbers? Well, like this...
This is `sec-thisone`!

Here is a proof that we want to be collapsed:

.. container:: theorem

    **Theorem.**

.. container:: toggle

    .. container:: proof

        **Proof** First we compute

        .. math::
            f(x) = W x + b
            :label: test2

    .. container:: header
    
        Toggle Proof

GraphViz
--------

.. graph:: foo

   "bar" -- "baz";

.. graphviz:: 

    digraph G {

            rankdir=LR
        splines=line
            
            node [fixedsize=true, label=""];

            subgraph cluster_0 {
            color=white;
            node [style=solid,color=blue4, shape=circle];
            x1 x2 x3;
            label = "layer 1 (Input layer)";
        }

        subgraph cluster_1 {
            color=white;
            node [style=solid,color=red2, shape=circle];
            a12 a22 a32;
            label = "layer 2 (hidden layer)";
        }

        subgraph cluster_2 {
            color=white;
            node [style=solid,color=seagreen2, shape=circle];
            O;
            label="layer 3 (output layer)";
        }

            x1 -> a12;
            x1 -> a22;
            x1 -> a32;
            x2 -> a12;
            x2 -> a22;
            x2 -> a32;
            x3 -> a12;
            x3 -> a22;
            x3 -> a32;

            a12 -> O
            a22 -> O
            a32 -> O
    }

References
----------

.. [TESTCIT2016] This is a great paper.
