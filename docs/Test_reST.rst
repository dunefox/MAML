Steinbruch
==========

Ok, here is a formula with a label

.. math::
    f(x) = W x + b
    :label: test

And we can refer to this equation as Eq. :eq:`test`. And this is an inline formula :math:`f(x)`.

We can immediately plot the function. Here, for :math:`W=2` and :math:`b=1`:

.. plot:: matplot.py
    :include-source:

For the plot we have used the code:

.. code-block:: python
    :linenos:

    def sum(a, b):
        return a + b

Or files can be sourced:

.. literalinclude:: matplot.py
    :linenos:

This is something taken from [TESTCIT2016]_

Here is some JavaScript

.. raw:: html
    :file: buttons.html

And here an interactive plot:

.. raw:: html
    :file: plot.html

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


References
==========

.. [TESTCIT2016] This is a great paper.
