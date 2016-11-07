.. _FirstSteps:

First steps: Linear Classifiers
===============================

Two traditional machine learning models for linear classifications are:

* The *Perceptron* and
* The *Adaptive Linear Neuron*.

The ideas for these two models go back to 

* McCulloh & Pitts (1943) :cite:`mcculloch_logical_1943`
* Rosenblatt (1957) :cite:`rosenblatt_perceptron_1957`
* Widrow (1960) :cite:`widrow_adapting_1960`

These models provide a good entry point also for modern machine learning algorithms as:

* They are very simple;
* but also generalizes to the modern *deep networks*.

This section is inspired by the introductory chapters of the book :cite:`raschka_python_2015`.

Binary Classification Problems
------------------------------

As an example of a typical of a binary classification problem let us consider:

* A sequence of :math:`N` data points :math:`x^{(i)}\in\mathbb R^2, 1\leq i\leq
  N`; 
* each having :math:`n` characteristic features
  :math:`x^{(i)}=(x^{(i)}_1,x^{(i)}_2,\ldots,x^{(i)}_3)`;
* and the task to assign to each element :math:`x^{(i)}` a label :math:`y^{(i)}\in\{-1,+1\}`;
* thereby dividing the data points into two classes labeled :math:`-1` and :math:`+1`. 
            
.. figure:: ./figures/linear_classification_data.png
    :width: 80%
    :align: center

    Labelled 2d example data points (:math:`n=2`) describing the sepal length and width of
    species of the iris flower. Here the class labels :math:`-1,+1` are encoded
    in the colors red and blue.

The goal of the classification problem is to find a function

.. math::
    f:\mathbb R^n \to \{-1,+1\}

that:

* predicts *accurately* the labels of pre-labeled training data
  :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M\leq N}`, i.e.,  for most indices
  :math:`i` it should hold :math:`f(x^{(i)})=y^{(i)}`;
* and *generalizes* well to unknown data.

The art of the game is to find sensible mathematical definitions for the vague
terms 'accurately' and 'generalizes' which allow to find in this sense optimal
functions :math:`f`.

One calls the classification problem linear if the data points of the two
classes can be separated by a hyper-plane. 

The following plot shows the data points of above with a possible hyper-plane
as decision boundary. 

.. figure:: ./figures/linear_classification_decission.png
    :width: 80%
    :align: center

    Decission boundaries for a possible classification function
    :math:`f`. The dots denote unknown data points and the crosses denote
    pre-labeled data points which were used to train the machine learning model
    in order to find an optimal :math:`f`.

Note that although the classification of the pre-labeled data seems to be
perfect, the classification of the unknown data is not. This may be due to the
fact that either:

* the data is simply not separable using just a hyper-plane, in which case one
  calls the problem 'non-linear classification problem',
* or possible errors in the pre-labeled data.

It is quite a typical situation that a perfect classification is not possible.
It is therefore important to specify mathematically in which sense we allow for
errors and what can be done to minimize them -- this will be encoded in the
mathematical sense given to 'accurately' and 'generalizes' as discussed above.

In this section we will specify two senses which lead to the model of the
Perceptron and Adaline.

Perceptron
^^^^^^^^^^

The Perceptron is a mathematical model inspired by a nerve cell:

.. figure:: ./figures/MultipolarNeuron.png
    :width: 80%
    :align: center

    A sketch of a neuron (`source <https://commons.wikimedia.org/wiki/File:Blausen_0657_MultipolarNeuron.png>`_).

The mathematical model can be sketched as follows:

.. figure:: ./figures/keynote/keynote.003.jpeg
    :width: 80%
    :align: center

The basic assumption in this simple mathematical model ist that the input
signals are weighted linearly. Hence, the classification function :math:`f` is
modeled by

.. math::
    f:\mathbb R^{n+1} &\to \{-1,+1\}\\
    x &\mapsto \sigma(w\cdot x)
    :label: eq-lin-model

.. note:: 

    * In the previous section the data points
      :math:`x^{(i)}=(x^{(i)}_1,\ldots,x^{(i)}_n)` were assumed to be from
      :math:`\mathbb R^n` and :math:`f` was assumed to be a :math:`\mathbb
      R^n\to\{-1,+1\}` function; 
    
    * A linear activation thus would amount to a function of the form

        .. math::
            f(x) = w \cdot x + b

      for weigths :math:`w\in\mathbb R^n` and threshold :math:`b\in\mathbb R`;

    * In the following absorb the threshold :math:`b` into the weight vector
      :math:`w` and therefore add a :math:`1` at the first position of
      all data vectors :math:`x^{(i)}`, i.e.

        .. math::
            \tilde x &= (1, x) = (1, x_1, x_2, \ldots, x_n),\\
            \tilde w &= (w_0, w) = (w_0, w_1, w_2, \ldots, w_n) = (b, w_1, w_2,\ldots, w_n);

    * so that 
        
        .. math::
            \tilde w\cdot \tilde x = w\cdot x + b

      where we will omit the overscript tilde in the future.
    
**Example:** 

    Let us pause and consider what such the simple model :eq:`eq-lin-model` is
    able to describe. The bitwise AND-gate is given by

    =======  =======  ======
    Input 1  Input 2  Output
    =======  =======  ======
    0        0        0
    0        1        0
    1        0        0
    1        1        1
    =======  =======  ======

    which we can represent as a graph similar to the iris data above:

    .. plot:: ./figures/python/and-gate.py
        :width: 80%
        :align: center

    The colors: red and blue denote the output values 0 or 1 of the AND-gate. Note
    that these two classes of data points can be well separated by a hyper-plane
    (in this case a line ☺). Hence, it is easy to find a  *good* weight vector :math:`w`. For instance:

    .. math::
        w 
        = 
        \begin{pmatrix}
            -1.5\\
            1\\
            1
        \end{pmatrix}.

    .. container:: toggle
            
        .. container:: header
        
            Homework

        .. container:: homework

            1. Check if :math:`f` in :eq:`eq:lin-model` with the weight vector
               given in :eq:`eq-weight-vector` decribes an AND-gate correctly and
               note that :math:`w` is by no means unique.
               
            2. Recall the geometric interpretation of the equation:

                .. math::
                    w\cdot x = 0

            3. Check all 16 bitwise logic gates and note which can be 'learned'
               by the model :eq:`eq-lin-model` and which not -- in the latter case, discuss
               why not.

Learning rule
"""""""""""""

Having specified the mathematical model :math:`f` in :eq:`eq-lin-londel` of the
neuron, the task is to learn a *good* weight vector in the sense discussed in
the previous section.

* This is now made by adjusting the weight vector :math:`w` according to the training data.
  :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M\leq N}`,
* in a way that minimizes the classification errors :math:`f(x^{(i)})\neq y^{(i)}`..

The algorithm by which the 'learning' is facilitated shall be called learning
rule and can be spell out as follows:

**INPUT:** Pre-labeled training data :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M\leq N}`

    **STEP 1:** Initialize the weight vector :math:`w` to zero or conveniently
    distributed random numbers.

    **STEP 2:** Pick a data point :math:`(x^{(i)},y^{(i)})` in the training samples at random:
    
        a) Compute the output

            :math:`y = f(x^{(i)})`

        b) Compare :math:`y` with :math:`y^{(i)}`:
        
            If :math:`y=y^{(i)}`, go back to **STEP 2**.

            Else, update the weight vector :math:`w` *appropriately*, and go back to **STEP 2**. 

The important step is the *update rule* which we discuss next.

Update rule
"""""""""""

First, we compute the difference between the correct label :math:`y^{(i)}`
given by the training data and the prediction :math:`y=f(x^{(i)})`:

.. math::
    \Delta = y^{(i)} - y

Second, we perform an update of the weight vector as follows:

.. math::
    w \mapsto \tilde w := w + \delta w
    :label: eq-update-weight

where

.. math::
    \delta w = \eta \, \Delta \, x^{(i)}.
    :label: eq-delta-weight

The parameter :math:`\eta\in\mathbb R^+` is called 'learning rate'.

Why could this update rule lead to a *good* choice of weights :math:`w`?

    Assume that in *STEP 2* b. of the learning rule identified
    a misclassification and calls the update rule. There are two possibilities:

    1. :math:`\Delta=2`: This means that the model predicted :math:`y=-1`
       although the correct label is :math:`y^{(i)}=1`. 
       
        * Hence, the by definition of :math:`f` in :eq:`eq-lin-model` the value
          of :math:`w\cdot x^{(i)}` is too low;
        * This can be fixed by adjusting the weights according to :eq:`eq-update-weight`
          and :eq:`eq-delta-weight`;
        * Next time this data point is examined one finds

            .. math::
                \tilde w \cdot x^{(i)} &= (w + \delta w)\cdot x^{(i)}\\
                                       &= w \cdot x^{(i)} + \eta \, \Delta \, (x^{(i)})^2
                                       &\geq w \cdot x^{(i)} 

          because, as :math:`\Delta > 0` and the square is non-negative,
          the last summand on the right is positive.
        * Hence, the new weight vector is changed in such a way that, next time, it is more
          likely that :math:`f` will predict the label of :math:`x^{(i)}` correctly.

    2. :math:`\Delta=-2`: This means that the model predicted :math:`y=1`
       although the correct label is :math:`y^{(i)}=-1`.  

        * By the same reasoning as in case 1. one finds: 
            
            .. math::
                \tilde w \cdot x^{(i)} &= (w + \delta w)\cdot x^{(i)}\\
                                       &= w \cdot x^{(i)} + \eta \, \Delta \, (x^{(i)})^2
                                       &\leq w \cdot x^{(i)} 

          because now we have :math:`\Delta < 0`, and again, the correction
          works in the right direction.

The model :eq:`eq-lin-model` for :math:`f` and this particular learning and
update rule defines the 'Perceptron'.

.. container:: toggle
        
    .. container:: header
    
        Homework

    .. container:: homework

        1. Implement a Perceptron for the 16 logical gates.
           
        2. Recall the geometric interpretation of the equation:

            .. math::
                w\cdot x = 0

        3. Check all 16 bitwise logic gates and note which can be 'learned'
           by the model :eq:`eq-lin-model` and which not -- in the latter case, discuss
           why not.


Adaline
^^^^^^^

Source code
-----------

[`Link <http://www.mathematik.uni-muenchen.de/~deckert/light-and-matter/teaching/WS1617/src/linear_classifiers/>`_] 


