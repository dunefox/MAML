.. highlight:: python
   :linenothreshold: 3

.. _FirstSteps:


First steps: Linear Classification
==================================

Two traditional machine learning models for linear classifications are:

* The *Perceptron* and
* The *Adaptive Linear Neuron (Adaline)*.

The ideas for these two models go back to:

    "A logical calculus of the ideas immanent in nervous activity." 
            
    -- McCulloh & Pitts :cite:`mcculloch_logical_1943`, 1943
    
    "The perception, a perceiving and recognizing automaton." 

    -- Rosenblatt :cite:`rosenblatt_perceptron_1957`, 1957
    
    "A adaptive 'Adaline' neuron using chemical memistors." 

    -- Widrow :cite:`widrow_adapting_1960`, 1957

These models provide a good entry point also for modern machine learning
algorithms as:

* They are very simple;
* but readily generalize to the concept of *deep networks*.

This section is in parts inspired by the introductory chapters of the book
*Python Machine Learning* by Sebastian Raschka :cite:`raschka_python_2015`.


Binary Classification Problems
------------------------------

As an example of a typical of a binary classification problem let us consider:

* A sequence of :math:`N`
* data points :math:`x^{(i)}\in\mathbb R^n, 1\leq i\leq N`; 

* each having :math:`n` characteristic features
  :math:`x^{(i)}=(x^{(i)}_1,x^{(i)}_2,\ldots,x^{(i)}_n)`;
* and the task to assign to each element :math:`x^{(i)}` a label :math:`y^{(i)}\in\{-1,+1\}`;
* thereby dividing the data points into two classes labeled :math:`-1` and :math:`+1`. 
   
.. figure:: ./figures/linear_classification_data.png
    :width: 80%
    :align: center

    Labelled :math:`n=2` dimensional example data points
    (:math:`x^{(i)}\in\mathbb R^2`) describing the sepal length and sepal
    width, i.e., :math:`x^{(i)}_1` and :math:`x^{(i)}_2`), respectively, of
    species of the Iris flower. The class label names 'setosa' and 'other',
    i.e., :math:`y^{(i)}=-1` and :math:`y^{(i)}=+1`, respectively, are encoded
    in the colors red and blue.

The goal of the classification problem is, given some pre-labeled training
data:

.. math::
   (x^{(i)},y^{(i)})_{1\leq i\leq M}, \qquad M< N 

to make the machine find a function

.. math::
    f:\mathbb R^n \to \{-1,+1\}

that:

* predicts *accurately* the labels of pre-labeled training data
  :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M}`, i.e.,  for *most* indices
  :math:`1\leq i\leq M` it should hold :math:`f(x^{(i)})=y^{(i)}`;
* and *generalizes* well the remaining data points :math:`x^{(i)}` for
  :math:`M>i\leq N` or even completely unknown data.

A general approach to this problem is to specify a space of candidates for
:math:`f`, the *hypotheses set*. Then the art of the game is to find sensible
and mathematical precise objects encoding the vague expressions 'accurately',
'most', and 'generalizes' and to find, in that sense, an optimal functions
:math:`f`. 

* Typically one tries to find an adequate coordinization of the hypotheses set,
  so that the search for an 'optimal' :math:`f` can be recast into a search
  for finitely many 'optimal' coordinates -- one often refers to the choice of
  coordinization and potential functions :math:`f` as the 'model' and to the
  particular coordinate as the 'parameters of the model';
* In which sense parameters are better or worse than others is usually encoded
  by a non-negative function on the set of possible parameters and the entire
  training data set, often called 'loss', 'regret', 'energy' or 'error'
  function;
* The search for optimal parameters is then recast into a search of minima of
  this loss function.

.. container:: definition

    **Definition (Classification Problem)** For :math:`n,c\in\mathbb N`, a set
    of :math:`c` labels :math:`I`, and a sequence :math:`x^{(i)}\in\mathbb
    R^n,y^{(i)}\in I`, :math:`1\leq i\leq M`, one calls the problem of finding
    a function :math:`f:\mathbb R^n\to I` such that :math:`f(x^{(i)})=y^{(i)}`
    for all :math:`1\leq i\leq N` an :math:`n`-dimensional
    *classification problem* with :math:`c` classes. 

    * The set :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M}`, is called *training data* 
      or *prelabeled data*.
    * In case, :math:`c=2` one referes to it as *binary classiciation problem*.
    * Furthermore, the problem is called a *linear classification problem* if the
      given data points :math:`x^{(i)}` can be separated according to their
      respective labels :math:`y^{(i)}` by means of hyperplanes. If this is not
      the case, one refers to the problem as *non-linear*.

The following plot shows the data points of the iris data set shown above with
a possible hyperplane as decision boundary between the two different classes.

.. _figLinBoundary:
.. figure:: ./figures/linear_classification_decission.png
    :width: 80%
    :align: center

    Decission boundaries for a possible classification function :math:`f`. The
    dots denote unknown data points, e.g., :math:`x^{(i)}` for :math:`M < i
    \leq N`, and the crosses denote pre-labeled data points, :math:`x^{(i)}`
    for :math:`1 \leq i \leq M`, which were used to train the model in order to
    find an optimal :math:`f`.

Note that in :numref:`figLinBoundary`, although the classification of the
pre-labeled data points (the crosses) seems to be perfect, the classification
of the unknown data (the dots) is not. This may be due to the following
reasons:

* the data is simply not separable using just a hyperplane, i.e., it is a
  non-linear classification problem,
* there are errors in the pre-labeled data,
* or the classifier function :math:`f` is not optimal yet.

It is quite a typical situation that a perfect classification is not possible.
It is therefore important to specify mathematically in which sense we allow for
errors and what can be done to minimize them -- this will be encoded in the
mathematical sense given to the expressions 'accurately' and 'generalizes' that
is usually encoded by means of choice in the loss function, as discussed above.

In the following we will specify two senses which lead to the model of the
Perceptron and Adaline.


Perceptron
----------

The first model we will take a look at is the so-called *Perceptron Model*.
It is a mathematical model inspired by a nerve cell depicted in
:numref:`figNerveCell`.

.. _figNerveCell:
.. figure:: ./figures/MultipolarNeuron.png
    :width: 80%
    :align: center

    A sketch of a neuron (`source <https://commons.wikimedia.org/wiki/File:Blausen_0657_MultipolarNeuron.png>`_).

The mathematical model can be sketched as in :numref:`figMathModelSketch`.

.. _figMathModelSketch:
.. figure:: ./figures/keynote/keynote.003.jpeg
    :width: 80%
    :align: center

    Sketch of the Perceptron Model.

* Let :math:`n\in\mathbb N` be the number of input signals;
* The input signals are given as a vector :math:`x\in\mathbb R^{n+1}`;
* These input signals are weighted by the weight vector :math:`w\in\mathbb R^{n+1}`,
* and then summed by means of the inner product :math:`w\cdot x`.
* The first coefficient in the input vector :math:`x` is always assumed to be
  one, and thus, the first coefficient in the weight vector :math:`w` is
  a threshold term, which renders :math:`w\cdot x` an *affine linear* as opposed to a 
  just *linear* map.
* Finally the signum function 

  .. math::
      \sigma(z) :=
      \begin{cases}
          +1 & \text{for } z\geq 0 \\
          +1 & \text{for } z< 0
      \end{cases}
  
  is employed to infer from :math:`w\cdot x\in\mathbb
  R` discrete class labels :math:`y\in\{-1,+1\}`.

This results in a hypothesis set of functions :math:`f_w`

.. math::
    f_w:\mathbb R^{n+1} &\to \{-1,+1\}\\
    x &\mapsto \sigma(w\cdot x)
    :label: eq-lin-model

parametrized by :math:`w\in\mathbb R^{n+1}`, where we shall often drop the subscript
:math:`w`. 

Since, our hypothesis set only contains linear functions, we may only expect it
to be big enough for linear (or approximately) linear classification problems.

.. note:: 

    * In the previous section the data points
      :math:`x^{(i)}=(x^{(i)}_1,\ldots,x^{(i)}_n)` were assumed to be from
      :math:`\mathbb R^n` and :math:`f` was assumed to be a :math:`\mathbb
      R^n\to\{-1,+1\}` function; 
    
    * Thus, an affine linear activation would amount to a function of the form

        .. math::
            f(x) = w \cdot x + b

      for weigths :math:`w\in\mathbb R^n` and threshold :math:`b\in\mathbb R`;

    * In the following absorb the threshold :math:`b` into the weight vector
      :math:`w` and therefore add the coefficient :math:`1` at the first position of
      all data vectors :math:`x^{(i)}`, i.e.

        .. math::
            \tilde x &= (1, x) = (1, x_1, x_2, \ldots, x_n),\\
            \tilde w &= (w_0, w) = (w_0, w_1, w_2, \ldots, w_n) = (b, w_1, w_2,\ldots, w_n);

    * so that 
        
        .. math::
            \tilde w\cdot \tilde x = w\cdot x + b.

    * Instead of an overset tilde, we will use the following convention to
      distinguish between vectors in :math:`\mathbb R^{n+1}` and :math:`\mathbb R^n`:

        .. math::
        	\mathbb R^{n+1} \ni x &= (1, \mathbf x) \in \mathbb R\times\mathbb R^n \\
        	\mathbb R^{n+1} \ni w &= (w_0, \mathbf w) \in \mathbb R\times\mathbb R^n
    
**Example:**  The bitwise AND-gate

    Let us pause and consider what such a simple model :eq:`eq-lin-model` is
    able to describe. This is a question of whether our hypothesis set is big
    enough to contain a certain function.

    The bitwise AND-gate operation is given by following table:

    =======  =======  ======
    Input 1  Input 2  Output
    =======  =======  ======
    0        0        0
    0        1        0
    1        0        0
    1        1        1
    =======  =======  ======

    * In order to answer the question, whether out hypothesis set is big enough
      to model the AND-gate, it is helpful to represent the above table as
      a graph similar to the iris data above. 
    
    * The features of each data point :math:`x^{(i)}` are the two input signals
      and the output value 0 and 1 are encoded by the class labels
      :math:`y^{i}\in\{-1,+1\}`.

    .. plot:: ./figures/python/and-gate.py
        :width: 80%
        :align: center

    * The colors: red and blue denote the output values 0 or 1 of the AND-gate.;
    * Note that the data points a linearly separable;
    * Note that these two classes of data points can be well separated by
      a hyperplane (in this case a 1d straight line). Hence, it is easy to find a  *good*
      weight vector :math:`w`. For instance:

    .. math::
        w 
        = 
        \begin{pmatrix}
            -1.5\\
            1\\
            1
        \end{pmatrix}.
        :label: eq-weight-vector

    .. container:: toggle
            
        .. container:: header
        
            Homework

        .. container:: homework

            1. Check if :math:`f` in :eq:`eq-lin-model` with the weight vector
               given in :eq:`eq-weight-vector` decribes an AND-gate correctly and
               note that :math:`w` is by no means unique.
               
            2. Give a geometric interpretation of the :math:`w`.

            3. Check all 16 bitwise logic gates and note which can be 'learned'
               by the model :eq:`eq-lin-model` and which not -- in the latter case, discuss
               why not.


Learning rule
~~~~~~~~~~~~~

Having settled for a hypothesis set such as the functions :math:`f_w`, 
:math:`w\in\mathbb R^{n+1}`, given in :eq:`eq-lin-model`,
the task is to learn a *good* parameters, i.e., in our case a *good* weight
vector :math:`w`, in the sense discussed in the previous section.

* This is now done by adjusting the weight vector :math:`w` appropriately
  depending on the training data :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M\leq
  N}`,
* in a way that minimizes the classification errors, i.e., the number of indices :math:`i` for which
  :math:`f(x^{(i)})\neq y^{(i)}`.

The algorithm by which the 'learning' is facilitated shall be called *learning
rule* and can be spell out as follows:

.. container:: algorithm

    **Algorithm: (Perceptron Learning Rule)** 

        **INPUT:** Pre-labeled training data :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M\leq N}`

            **STEP 1:** Initialize the weight vector :math:`w` to zero or conveniently
            distributed random coefficients.

            **STEP 2:** Pick a data point :math:`(x^{(i)},y^{(i)})` in the training samples at random:
    
                i) Compute the output

                    :math:`y = f(x^{(i)})`

                ii) Compare :math:`y` with :math:`y^{(i)}`:
        
                     If :math:`y=y^{(i)}`, go back to **STEP 2**.

                     Else, update the weight vector :math:`w` *appropriately* according to an *update rule*, 
                     and go back to **STEP 2**. 

The following sketch is a visualization of the feedback loop for the learning rule:

.. figure:: ./figures/keynote/keynote.004.jpeg
    :width: 80%
    :align: center

The important step is the *update rule* which we discuss next.


Update rule
~~~~~~~~~~~

Let us spell out a possible update rule and then discuss why it does what we want:

    First, we compute the difference between the correct label :math:`y^{(i)}`
    given by the training data and the prediction :math:`y=f(x^{(i)})`:

    .. math::
        \Delta^{(i)} := y^{(i)} - y
        :label: eq-delta

    Second, we perform an update of the weight vector as follows:

    .. math::
        w \mapsto w^{\text{new}} := w + \delta w
        :label: eq-update-weight

    where

    .. math::
        \delta w := \eta \, \Delta^{(i)} \, x^{(i)}.
        :label: eq-delta-weight

    The parameter :math:`\eta\in\mathbb R^+` is called 'learning rate'.

Why does this update rule lead to a *good* choice of weights :math:`w`?

    Assume that in **STEP 2** b. of the learning rule identified
    a misclassification and calls the update rule. There are two possibilities:

    1. :math:`\Delta=2`: This means that the model predicted :math:`y=-1`
       although the correct label is :math:`y^{(i)}=1`. 
       
        * Hence, by definition of :math:`f` in :eq:`eq-lin-model` the value
          of :math:`w\cdot x^{(i)}` is too low;
        * This can be fixed by adjusting the weights according to :eq:`eq-update-weight`
          and :eq:`eq-delta-weight`;
        * Next time when this data point is examined one finds

            .. math::
                w^{\text{new}} \cdot x^{(i)} &= (w + \delta w)\cdot x^{(i)}\\
                                       &= w \cdot x^{(i)} + \eta \, \Delta \, (x^{(i)})^2 \\
                                       &\geq w \cdot x^{(i)} 

          because, as :math:`\Delta > 0` and the square is non-negative,
          the last summand on the right is positive.
        * Hence, the new weight vector is changed in such a way that, next time, it is more
          likely that :math:`f` will predict the label of :math:`x^{(i)}` correctly.

    2. :math:`\Delta=-2`: This means that the model predicted :math:`y=1`
       although the correct label is :math:`y^{(i)}=-1`.  

        * By the same reasoning as in case 1. one finds: 
            
            .. math::
                w^{\text{new}} \cdot x^{(i)} &= (w + \delta w)\cdot x^{(i)}\\
                                       &= w \cdot x^{(i)} + \eta \, \Delta \, (x^{(i)})^2 \\
                                       &\leq w \cdot x^{(i)} 

          because now we have :math:`\Delta < 0`, and again, the correction
          works in the right direction.

The model :eq:`eq-lin-model` for :math:`f`, i.e., hypothesis set, and this
particular learning and update rule is what defines the 'Perceptron'.


Convergence
~~~~~~~~~~~

Now that we have a heuristic understanding why the learning and update rule
chosen for the Perceptron works, we have a look at what can be said
mathematically; see :cite:`varga_neural_1996` for a more detailed discussion.

First let us make precise what we mean by 'linear separability' in our setting:

.. container:: definition

    **Definition: (Linear seperability)** Let :math:`A,B` be two sets in :math:`\mathbb R^n`. Then:

    1. :math:`A,B` are called *linearly seperable* if there is a
    
        :math:`w\in\mathbb R^{n+1}` such that

            .. math:: 
                \forall\, a\in A,\, b\in B: 
                \quad 
                w\cdot a \geq 0 \quad \wedge 
                \quad
                w\cdot b < 0.

    2. :math:`A,B` are called *absolutely linearly seperable* if there is a

        :math:`w\in\mathbb R^{n+1}` such that

            .. math:: 
                \forall\, a\in A,\, b\in B: 
                \quad 
                w\cdot a > 0 \quad \wedge 
                \quad
                w\cdot b < 0.

The learning and update rule algorithm of the Perceptron can be formulated in
terms of the following algorithm:

.. container:: algorithm

    **Algorithm: (Perceptron Learning and Update Rule)** 

        **PREP:** 
        
            Prepare the training data :math:`(x^{(i)},y^{(i)})_{1\leq
            i\leq M}`. Let :math:`A` and :math:`B` be the sets of elements
            :math:`x^{(i)}\in\mathbb R^{n+1}=(1,\mathbf x^{(i)})` whose class labels
            fulfill :math:`y^{(i)}=+1` and :math:`y^{(i)}=-1`, respectively.

        **START:** 
        
            Initialize the weight vector :math:`w^{(0)}\in\mathbb R^{n+1}` with
            random numbers and set :math:`t\gets 0`.

        **STEP:** 
        
            Choose :math:`x\mathbb \in A,B` at random:

            * If :math:`x\in A, w^{(t)}\cdot x \geq 0`: goto **STEP**.
            * If :math:`x\in A, w^{(t)}\cdot x < 0`: goto **UPDATE**.
            * If :math:`x\in B, w^{(t)}\cdot x \leq 0`: goto **STEP**.
            * If :math:`x\in B, w^{(t)}\cdot x > 0`: goto **UPDATE**.

        **UPDATE:** 
        
            * If :math:`x\in A`, then set :math:`w^{(t+1)}:=w^{(t)} + x`,
              increment :math:`t\gets t+1`, and goto **STEP**.
            * If :math:`x\in B`, then set  :math:`w^{(t+1)}:=w^{(t)} - x`,
              increment :math:`t\gets t+1`, and goto **STEP**.

* Note that for an implementation of this algorithm we will also need an exit
  criterion so that the algorithm does not run forever. 

* This is usually done by specifying how many times the entire training set
  is run through **STEP**, a number which is often referred to as number of
  *epochs*.

* Note further, that for sake of brevity , the learning rate :math:`\eta` was
  chosen to equal :math:`1/2`; compare to :eq:`eq-delta-weight`.

Frank Rosenblatt already showed convergence of the algorithm above in the case of finite and linearly separable training data:

.. container:: theorem

    **Theorem: (Perceptron convergence)**

        Let :math:`A,B` be finite sets and linearly seperable, then the number of updates performed by the Perceptron algorithm stated above is finite.

    .. container:: toggle
            
        .. container:: header
        
            Proof

        .. container:: proof

            * As a first step, we observe that since :math:`A,B` are finite
              sets that are linear seperable, they are also absolutely
              seperable due to:

            .. container:: theorem

                **Proposition:**

                Let :math:`A,B` be finite sets of :math:`\mathbb R^{n+1}`:
                :math:`A,B` are linearly seperable :math:`\Leftrightarrow`
                :math:`A,B` are absolutely linearly seperable.
                
                .. container:: toggle
                        
                    .. container:: header
                    
                        Proof

                    .. container:: proof

                        Homework.

            * Furthermore, we observe that without restriction of generality
              (WLOG) we may assume the vectors :math:`x\in A\cup B` to be
              normalized because
              
                .. math::
                    w\cdot x > 0 \,  \vee \, w\cdot x < 0 
                    \Leftrightarrow 
                    w\cdot \frac{x}{\|x\|} > 0 \,  \vee \, w\cdot \frac{x}{\|x\|} < 0.

              Note that this means that for such :math:`x`, :math:`x_0` does
              not equal one in general, and hence, we break our convention.
              However, the reason for this convention was to ensure that
              :math:`x\mapsto w\cdot x` is an affine linear map with a
              potential bias term :math:`w^0 x^0`.  As long as :math:`x^0\neq
              0` this is the case and any necessary scaling will be encoded
              into the choice of :math:`w^0` during training. 

            * Let us define :math:`T=A\cup (-1)B`, i.e., :math:`T` is the union
              of :math:`A` and the element of :math:`B` times :math:`(-1)`.

            * Since :math:`A,B` absolutely linearly seperable there is a
              :math:`w^*\in\mathbb R^{n+1}` such that for all :math:`x\in T`

                .. math::
                    w^{*}\cdot x > 0.
                    :label: eq-abs-lin

              And moreover, we also may WLOG assume that :math:`w^{*}` is normalized.

            Let us assume that some time after the :math:`t`-th update a point
            :math:`x\in T` is picked in **STEP** that leads to a
            misclassification

                .. math::
                    w^{(t)} \cdot x < 0

            so that **UPDATE** will be called which updates the weight vector according to

                .. math::
                    w^{(t+1)} := w^{(t)} + x.

            Note that both cases of **UPDATE** are treated with this update since in the definition of :math:`T` we have already included the 'minus' sign.

            Now in order to infer a bound on the number of updates :math:`t` in the
            Perceptron algorithm above, consider the quantity

                .. math::
                    1\geq \cos \varphi = \frac{w^{*}\cdot w^{(t+1)}}{\|w^{(t+1)}\|}.
                    :label: eq-denum

            To bound this quantity also from below, we consider first:

                .. math::
                    w^{*}\cdot w^{(t+1)} = w^{*}\cdot w^{(t)} + w^{*}\cdot x.

            Thanks to :eq:`eq-abs-lin` and the finiteness of :math:`A,B`, we know that

                .. math::
                    \delta := \min\{w^*\cdot x \,|\, x \in T\} > 0.
                    :label: eq-delta

            This facilitates the estimate
                
                .. math::
                    w^{*}\cdot w^{(t+1)} \geq  w^{*}\cdot w^{(t)} + \delta,

            which, by induction, gives

                .. math::
                    w^{*}\cdot w^{(t+1)} \geq  w^{*}\cdot w^{(0)} + (t+1)\delta.
                    :label: eq-ing-1

            Second, we consider the denumerator of :eq:`eq-denum`:

                .. math::
                    \| w^{(t+1)} \|^2 = \|w^{(t)}\|^2 + 2 w^{(t)}\cdot x + \|x\|^2.

            Recall that :math:`x` was misclassified by weight vector :math:`w^{(t)}` so that :math:`w^{(t)}\cdot x<0`. This yields the estimate
                
                .. math::
                    \| w^{(t+1)} \|^2 \leq  \|w^{(t)}\|^2 + \|x\|^2.

            Again by induction, and recalling the assuption that :math:`x` was normalized, we get:

                .. math::
                    \| w^{(t+1)} \|^2 \leq  \|w^{(0)}\|^2 + (t+1).
                    :label: eq-ing-2

            Both bounds, :eq:`eq-ing-1` and :eq:`eq-ing-2`, together with
            :eq:`eq-denum`, give rise to the inequalities

                .. math::
                   1 \geq \frac{w^{*}\cdot w^{(t+1)}}{\|w^{(t+1)}\|} \geq\frac{w^{*}\cdot w^{(0)} + (t+1)\delta}{\sqrt{\|w^{(0}\|^2 + (t+1)}}.
                   :label: eq-fin-est

            The right-hand side would grow as :math:`O(\sqrt t)` but has to be
            smaller one. Hence, :math:`t`, i.e., the number of updates, must be
            bounded by a finite number.

.. container:: toggle
        
    .. container:: header
    
        Homework

    .. container:: homework

        1. What is the geometrical meaning of :math:`\delta` in :eq:`eq-delta`
           in the proof above?

        2. Consider the case :math:`w^{(0)}=0` and give an upper bound on the
           maximum number of updates.

        3. Carry out the analysis above including an arbitrary learning rate
           :math:`\eta`. How does :math:`\eta` influence the number of
           updates?

Finally, though this result is reassuring it needs to be emphasized that it is
rather academic. 

* The convergence theorem only holds in the case of linear separability of the
  test data, which in most interesting cases is not given.


Python implementation 
~~~~~~~~~~~~~~~~~~~~~~

Next, we discuss an Python implementation of the Perceptron discussed above.

* The mathematical model of the function :math:`f`, i.e., the hypothesis set,
  the learning and update rule will be implemented as a Python class::
      
      class Perceptron:

          def __init__(self, num):
              '''
              initialize class for `num` input signals
              '''

              # weights of the perceptron, initialized to zero
              # note the '1 + ' as the first weight entry is the threshold
              self.w_ = np.zeros(1 + num)

              return

* The constructor ``__init__`` takes as argument the number of input signals
  ``num`` and initializes the variable ``w_`` which will be used to store the
  weight vector :math:`w\in\mathbb R^{n+1}` where :math:`n=` ``num``.

  The constructor is called when an object of the ``Perceptron`` class is
  created, e.g., by::

    ppn = Perceptron(2)

  In this example, it initializes a Perceptron with :math:`n=2`.

* The first method ``activation_input`` of the Perceptron class takes as
  argument an array of data points ``X``, i.e., :math:`(x^{(i)})_i`, and
  returns the array of input activations :math:`w\cdot x^{(i)}` for all
  :math:`i` using the weight vector :math:`w` stored in variable ``w_``::

    def activation_input(self, X):
        '''
        calculate the activation input of the neuron
        '''
        return np.dot(X, self.w_[1:]) + self.w_[0]

* The second method ``classify`` takes again an array of data points ``X``,
  i.e., :math:`(x^{(i)})_i` as argument. It uses the previous method
  ``input_activation`` to compute the input activations :math:`(w\cdot
  x^{(i)})_i` and then applies the signum function to the values in the arrays::

    def classify(self, X):
        '''
        classify the data by sending the activation input through a step function
        '''
        return np.where(self.activation_input(X) >= 0.0, 1, -1)

  This method is the implantation of the function :math:`f` in :eq:`eq-lin-model`.

* Finally, the next method implements the learning and update rule::

    def learn(self, X_train, Y_train, eta=0.01, epochs=10):
        '''
        fit training features X_train with labels Y_train according to learning rate
        `eta` and total number of epochs `epochs` and log the misclassifications in errors_
        '''
        
        # reset internal list of misclassifications for the logging
        self.train_errors_ = [] 

        # repeat `epochs` many times
        for _ in range(epochs):
            err = 0
            # for each pair of features and corresponding label
            for x, y in zip(X_train, Y_train):
                # compute the update for the weight coefficients
                update = eta * ( y - self.classify(x) )
                # update the weights
                self.w_[1:] += update * x
                # update the threshold
                self.w_[0] += update
                # increment the number of misclassifications if update is not zero
                err += int(update != 0.0)
            # append the number of misclassifications to the internal list
            self.train_errors_.append(err)
    
        return
        
  * It takes as input arguments the training data
    :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M}` in form of two arrays ``X_train``
    and ``Y_train``, and furthermore, the learning rate ``eta``, i.e.,
    :math:`\eta`, and an additional number called ``epochs``. 
  
  * The latter number ``epochs`` specifies how many times the learning rule
    runs over the whole training data set -- see the ``for`` loop in line
    number 11.

  * In the body of the first ``for`` loop a variable `err` is set to zero and
    a second ``for`` loop over set of training data points is carried out.

  * The body of the latter ``for`` loop implement the update rule
    :eq:`eq-delta`-:eq:`eq-delta-weight`.

  * Note that there are two types of updates, i.e., lines 18 and 20. This is
    due to the fact that above we used the convention that the first
    coefficient of :math:`x` was fixed to one in order to keep the notation
    slim.

  * In line 22 ``err`` is incremented each time a misclassification occurred.
    The number of misclassification per epoch is then append to the list
    ``train_errors``.

After loading to training data set :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M}`
into the two arrays ``X_train`` and ``Y_train`` the Perceptron can be
initializes and trained as follows::

    ppn = Perceptron(X.shape[1])
    ppn.learn(X_train, Y_train, eta=0.1, epochs=100)

Find the full implementation here: [`Link <https://github.com/dirk-deckert/MAML/blob/master/src/first_steps/001_iris_perceptron.ipynb>`_] 

.. container:: toggle
        
    .. container:: header
    
        Homework

    .. container:: homework

        Have a look at the Perceptron implementation (link given above):
                   
        a. What effect does the learning rate have? Examine a situation is
        which the learning rate is too high and too low and discuss both cases.
        
        b. What happens when the training data cannot be separated by
        a hyperplane? Examine problematic situation and discuss these -- for
        example, by generating fictitious data points.

        c. Note that the instant all training data was classified correctly
        the Perceptron stops to update the weight vector. Is this a feature or
        a bug?

        d. Discuss the dependency of the learning success on the order in which
        the training data is presented to the Perceptron. How could the
        dependency be suppressed?

        Find an implementation showing different learning behaviors here: [`Link <https://github.com/dirk-deckert/MAML/blob/master/src/first_steps/002_iris_perceptron_convergence.ipynb>`_] 


Problems with the Perceptron
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* As discussed, the convergence of the Perceptron algorithm is only guaranteed
  in the case of linearly separable test data. 
* If linear separability is not provided, in each epoch will be at least one
  update that will result in an oscillatory behavior in the chosen :math:`w`.
* Thus, in general we need a good exit criterion for the algorithm to
  bound the maximum number of updates.
* The updates stop the very instant the entire test data is classified correctly,
  which might lead to poor generalization properties of the resulting
  classifier to unknown data.

.. container:: toggle
        
    .. container:: header
    
        Homework

    .. container:: homework

        Implement training scenarios for the Perceptron in which you can
        observe the qualitative behavior described above, i.e., the possible
        oscillations and the abrupt stop in training.


Adaline
-------

* The Adaline algorithm will overcome some of the short-comings of the one of Perceptron.
* The basic design is almost the same:

    .. figure:: ./figures/keynote/keynote.005.jpeg
        :width: 80%
        :align: center

* The first difference w.r.t. to the Perceptron is the additional activation
  function :math:`\alpha`. We shall call :math:`w\cdot x` *activation input* and
  :math:`\alpha(w\cdot x)` *activation output*.

* We will discuss different choices of activation functions later. For now let
  us simply use: 
  
    .. math::
        \alpha: \mathbb R &\to \mathbb R \\
        \alpha &\mapsto \alpha(z):=z.
        :label: eq-alpha-z

* The second difference is that the activation output is used as in feedback
  loop for the update rule.

* The advantage is that, provided :math:`\alpha:\mathbb R\to\mathbb R` is
  regular enough, we may make use of analytic optimization theory in order to find an in some sense 'optimal' choice of weights :math:`w\in\mathbb R^{n+1}`.

* This was not possible in the case of the Perceptron because the signum
  function is not differentiable.


Update rule
~~~~~~~~~~~

* Recall that an 'optimal' choice of weights :math:`w\in\mathbb R^{n+1}` should fulfill two properties:

    1. It should 'accurately' classify the training data :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M}`,
    2. and it should 'generalize' well unknown data.

* In order to make use of analytic optimization theory, one may attempt to
  encode a measure of the optimality of weights w.r.t. these two properties in
  form of a function that attains smaller and smaller values the better the
  weights fulfill these properties.

* This function is called many names, e.g., 'loss', 'regret', 'cost', 'energy',
  or 'error' function. We will use the term 'loss function'.

* Of course, depending on the classification task, there are many choices. Maybe
  one of the simplest examples is:

    .. math::
        L:\mathbb R^{n+1} &\to \mathbb R^+_0 \\
        w &\mapsto L(w) 
        := 
        \frac12 \sum_{i=1}^M \left(y^{(i)} - \alpha(w\cdot x^{(i)})\right)^2,
        :label: eq-L

  which is the accumulated squared euclidean distance between the particular
  labels of the test data :math:`y^{(i)}` and the corresponding prediction
  :math:`\alpha(w\cdot x^{(i)})` given by Adaline for the current weight vector
  :math:`w`.

* Note that the loss function depends not only on :math:`w`, but also on the
  entire training data set :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M}`. The
  latter, however, is assumed to be fixed which is why the dependence of
  :math:`L(w)` on it will be suppressed in out notation.

* From its definition the loss function in :eq:`eq-L` has the desired
  property that it grows and decreases whenever the number of
  misclassification grows or decreases, respectively.

* Furthermore, it does so smoothly, which allows for the use of analytic
  optimization theory.
        
.. container:: toggle
        
    .. container:: header
    
        Homework

    .. container:: homework

        What criteria should a general loss function :math:`L(w)` fulfill?


Learning and update rule
~~~~~~~~~~~~~~~~~~~~~~~~

* Having encoded the desired properties of 'optimal' weights :math:`w\in\mathbb
  R^{n+1}` as a global minimum of the function :math:`L(w)`, the only
  task left to do is to find this global minimum.

* Depending on the function :math:`L(w)`, i.e., on the training data, this task
  may be arbitrarily simple or difficult.

* Consider the following heuristics in order to infer a possible learning
  strategy:

    * Say, we start with a weight vector :math:`w\in\mathbb R^{n+1}` and want
      to make an update

        .. math::
            w\mapsto w^{\text{new}}:=w + \delta w

      in a favourable direction :math:`\delta w\in\mathbb R^{n+1}`.

    * An informal Taylor expansion of :math:`L(w^{\text{new}})` reveals

        .. math::
            L(w^{\text{new}}) = L(w) + \frac{\partial L(w)}{\partial w} \delta w + O(\delta w^2).

    * In order to make the update 'favourable' we want that :math:`L(w^{\text{new}})\leq L(w)`.

    * Neglecting the higher orders, this would mean:

        .. math::
            \frac{\partial L(w)}{\partial w} \delta w < 0.
            :label: eq-L-diff

    * In order to get rid of the unknown sign of :math:`\frac{\partial L(w)}{\partial
      w}` we may choose:

        .. math::
            \delta w := - \eta \frac{\partial L(w)}{\partial w} 
            :label: eq-L-deltaw

      for some learning rate :math:`\eta\in\mathbb R^+`. 
      
    * Then, for the choice :eq:`eq-L-deltaw` the linear order
      :eq:`eq-L-diff` becomes negative and we note that

        .. math::
            L(w^{\text{new}}) 
            = 
            L(w) - \eta \left(\frac{\partial L(w)}{\partial w}\right)^2 + O_{\delta w\to 0}(\delta w^2).

      Hence, the update may work to decrease the value of :math:`L(w)` – at
      least in the linear order of perturbation.

Concretely, for our case we find:

    .. math::
        \frac{\partial L(w)}{\partial w} 
        = 
        -\sum_{i=1}^M 
        \left(
            y^{(i)}-\alpha(w\cdot x^{(i)})
        \right)
        \alpha'(w\cdot x^{(i)}) \, x^{(i)},
        :label: eq-dL-dw

where :math:`\alpha'` denotes the derivative of :math:`\alpha`. Here the
notation :math:`\partial/\partial w` denotes the gradient

.. math::
    \frac{\partial}{\partial w} =
    \left(\frac{\partial}{\partial w_j}\right)_{0\leq j\leq n}

and :eq:`eq-dL-dw` makes sense as :math:`x^{(i)}` on the right-hand side is a
vector in :math:`\in\mathbb R^{n+1}`.

In conclusion, we may formulate the Adaline algorithm as follows:

.. container:: algorithm

    **Algorithm: (Adaline Learning and Update Rule)** 

        **INPUT:** Pre-labeled training data :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M\leq N}`

            **STEP 1:** Initialize the weight vector :math:`w` to zero or conveniently
            distributed random coefficients.

            **STEP 2:** For a certain number of epochs:
    
                i) Compute :math:`L(w)`

                ii) Update the weights :math:`w` according to

                     .. math::
                         w \mapsto w^{\text{new}} := w + \eta \sum_{i=1}^M 
                         \left(
                             y^{(i)}-\alpha(w\cdot x^{(i)})
                         \right)
                         \alpha'(w\cdot x^{(i)}) x^{(i)} 
                         :label: eq-adaline-update
    
.. container:: toggle

    .. container:: header

        Homework

    .. container:: homework

        1. Prove that even in the linearly seperable case, the above Adaline
           algorithm does not need to converge. Do this by constructing
           a simple example of training data and a special choice of learning rate
           :math:`\eta`.

        2. What is the influence of large or small values of :math:`\eta`?

        3. Discuss the advantages/disadvantages of immediate weight updates
           after misclassification as it was the case for the
           Perceptron and batch updates as it is the case for Adaline.

                   
Python implementation 
~~~~~~~~~~~~~~~~~~~~~~

As we have already noted, the Adaline learning rule is the same as the one of
the Perceptron. Hence, for linear activation :math:`\alpha(z)=z`, we only need
to change the learning rule implemented in the method ``learn`` of the
``Perceptron`` class. The ``Adaline`` class can therefore we created as
follows::

    class Adaline(Perceptron):

        def learn(self, X_train, Y_train, eta=0.01, epochs=1000):
            '''
            fit training data according to eta and n_iter
            and log the errors in errors_
            '''

            # we initialize two list, each for the misclassifications and the cost function
            self.train_errors_ = []
            self.train_loss_ = []

            # for all the epoch
            for _ in range(epochs):
                # classify the training features
                Z = self.classify(X_train)
                # count the misclassifications for the logging
                err = 0
                for z, y in zip(Z, Y_train):
                    err += int(z != y)
                # ans save them in the list for later use
                self.train_errors_.append(err)
                
                # compute the activation input of the entire training features
                output = self.activation_input(X_train)
                # and then the deviation from the labels
                delta = Y_train - output
                # the following is an implementation of the Adaline update rule
                self.w_[1:] += eta * X_train.T.dot(delta)
                self.w_[0] += eta * delta.sum()
                # and finally, we record the loss function
                loss = (delta ** 2).sum() / 2.0
                # and save it for later use
                self.train_loss_.append(loss)

            return

* Line 1 defines the ``Adaline`` class and a child of the ``Perceptron`` one.
  It thus inherits all the methods and variables of the ``Perceptron`` class.
* Line 11 introduces a similar variable as ``train_errors`` that will store the value of the loss function 
  per epoch.
* Line 14 is again the ``for`` loop over the epochs:
* In line 16 the classification of all training data points is conducted.
* Lines 17-22 only count the number of misclassification which is then appended
  to the list ``train_errors_``.
* The update rule is implemented in Lines 24-30. First, the input activation of
  all the training data is computed and the array ``delta`` stores the set
  :math:`(y^{(i)}-w \cdot x^{i)})`.
* This ``delta`` array is then used to compute the updated weight vector stored in ``w_`` in lines 29-30.
* The last two lines in this ``for`` loop compute the loss value for this epoch
  and store it in the list ``train_loss_``.


Find the full implementation here: [`Link <https://github.com/dirk-deckert/MAML/blob/master/src/first_steps/004_iris_adaline.ipynb>`_] 


Learning behavior
~~~~~~~~~~~~~~~~~

* We have introduced the learning rate :math:`\eta` in an ad-hoc fashion;
* Not even in the linear separable case, we may expect convergence of the Adaline algorithm;
* We can only expect to find a good approximation of the optimal choice of
  weights :math:`w\in\mathbb R^{n+1}`;
* But the approximation will depend on the choice of the learning rate parameter.

In the figure below, the learning rate :math:`\eta` was chosen too large.
Instead of approximating the minimum value, the gradient descent algorithm even
diverges.
    
.. plot:: ./figures/python/learning_rate_too_large.py
    :width: 80%
    :align: center

In the next figure, the learning rate :math:`\eta` has been chosen too small.
In case, the loss function has other local minima, the initial weight vector
:math:`w` is coincidently chosen near such a local minimum, and the learning
rate is too small, the gradient descent algorithm will converge too the nearest
local minimum instead of the global minimum.
    
.. plot:: ./figures/python/learning_rate_too_small.py
    :width: 80%
    :align: center

In the special case of a linear activation :math:`\alpha(z)=z` and a quadratic
loss function such as :eq:`eq-L`, which we also used in our Python
implementation, such a behavior cannot occur due to convexity. For more general
activations :math:`\alpha(z)` and loss fuctions :math:`L(w)` that we will
discuss later there are often non-trivial landscape of local minima.

.. container:: toggle
            
    .. container:: header
        
        Homework

    .. container:: homework

        Prove that for a quadratic loss function as given in :eq:`eq-L` and the
        choice of a linear activation :math:`\alpha(z)=z`, the corresponding
        loss function is convex independently of the training data. What does
        that mean for the Adaline update rule?

Here is another bad scenario where we see that the gradient descent algorithm
does not converge:

.. plot:: ./figures/python/learning_rate_no_convergence.py
    :width: 80%
    :align: center

Many improvements can be made with respect to the gradient descent algorithms
which tend to work well in different situations. A behavior such as in the
scenario above can be tempered by adapting the learning rate. For instance, by
choosing:

.. math::
    \eta = \frac{c_1}{c_2+t},

where :math:`c_1,c_2\in\mathbb R^+` are two constants and :math:`t` is the
number of updates. 


Here is a nice overview on
popular optimization algorithms used in machine learning: 
`[An overview of gradient descent optimization algorithms] <http://sebastianruder.com/optimizing-gradient-descent/>`_ by Sebastian Ruder.


Stadardization of training data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As can be seen from :eq:`eq-L-deltaw`, the learning rate :math:`\eta` controls
the magnitude of the update of the weight vector. The scale on which the
learning rate controls the update is given by the factor :math:`\frac{\partial
L(w)}{\partial w}` which by definition of :math:`L(w)` in :eq:`eq-L` is given
in terms of a sum of :math:`N` summands. Hence, if the size of the training set
varies the respective scales of the learning rate will in general not be
comparable. To reduce the dependence of the learning rate scale on the size of
the training data, it a good idea to replace :math:`L(w)` by the average:

.. math::
    L:\mathbb R^{n+1} &\to \mathbb R^+_0 \\
    w &\mapsto L(w) 
    := 
    \frac{1}{2M} \sum_{i=1}^M \left(y^{(i)} - \alpha(w\cdot x^{(i)})\right)^2,

Furthermore, the learning rate will depend on the fluctuations in the features
of your training data set. In order, to have comparable results it is therefore
a good idea to normalize the training data.  A standard procedure to do this is
to transform the training data according to the following map

.. math::
    x^{(i)} \mapsto \widetilde x^{(i)} := \frac{x^{(i)} - \overline{x}_M}{\sigma},

where 

.. math::
    \overline{x}_M:=\frac1M\sum_{i=1}^M x^{(i)}
    
is the empirical average and

.. math::
    \sigma:=\sqrt{\frac1M\sum_{i=1}^M (x^{(i)} - \overline{x}_M)}
    :label: eq-sigma

is the standard variation. This procedure is called *standardization* of the
training data.

.. container:: toggle
            
    .. container:: header
        
        Homework

    .. container:: homework

        A little side remark from statistics: For small samples :eq:`eq-sigma`
        underestimates the standard deviation on avarage. A better estimate is
        therefore:

        .. math::
            s_M:=\sqrt{\frac{1}{M-1}\sum_{i=1}^M (x^{(i)} - \overline x_M)}

        Prove that for independent identically distributed random variables we
        get :math:`E(s_M^2)=\sigma^2` thanks to the pecuiliar factor
        :math:`(M-1)^{-1}`.  However, in general our training samples will be
        large enough that this goes unnoticed.


Online learning versus batch learning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* The learning and update rule of the Perceptron and Adaline have a crucial difference:
    
    * The Perceptron updates its weights according to inspection of a single
      data point :math:`(x^{(i)},y^{(i)})` of the training data
      :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M}` -- this is usually referred to
      as 'online' (in the sense of 'real-time') or 'stochastic' (in the sense
      that training data points are chosen at random) learning. 
      
    * The Adaline conducts an update after inspecting the whole training data
      :math:`(x^{(i)},y^{(i)})_{1\leq i\leq M}` by means of computing the
      gradient of the loss function :eq:`eq-L` that depends on the entire set
      of training data -- this is usually referred to as 'batch' learning (in
      the sense that the whole batch of training data is used to compute an
      update of the weights).

* While online learning may have a strong dependence on the sequence of training
  data points presented to the learner and produces updates that may be extreme
  for extreme outliers, batch learning averages out the updates but therefore
  is usually computationally very expensive.

* Clearly, we can also make Adaline become an online learner by receptively
  presenting it training data consisting only of one randomly chosen point. This
  method is called *stochastic gradient descent*.

* In turn, we can make the Perceptron become a batch learner, simply by
  computing all the update per element in the entire training data set,
  computing the average update, and then performing a single update with the
  average.

* A compromise between the two extremes, online and batch learning, is the so-called 'mini-batch'
  learning. 
  
    * The entire batch of training data :math:`I=(x^{(i)},y^{(i)})_{1\leq i\leq
      M}` is then split up into disjoint mini-batches
      :math:`I_k:=(x^{(i)},y^{(i)})_{i_{k-1}\leq i\leq i_{k}}` for an appropriate
      strictly increasing sequence of indices :math:`(i_k)_k` such that
      :math:`I=\bigcup_k I_k`. 
    * For each mini-batch :math:`I_k` the mini-batch loss function is computed
        
        .. math::
            L(w) := \frac12 \sum_{i=i_{k-1}}^{i_k-1} \left(y^{(i)} - \alpha(w\cdot x^{(i)})\right)^2,

      and the update of the weight vector :math:`w` is performed accordingly.

* For appropriate chosen mini-batch sizes this mini-batch learning rule often
  proves to be faster than online or batch learning.
    
.. container:: toggle
        
    .. container:: header
    
        Homework

    .. container:: homework

        Implement the Pereceptron and Adaline as online, batch, and mini-batch
        learners and study resepctive learning success for the same set of training
        data.
