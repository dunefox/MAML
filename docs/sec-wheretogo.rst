
Where to go from here?
======================

The step from the Perceptron to Adaline mainly brings two advantages:

1. We may make use of analytic optimization theory;
2. We may encode what we mean by 'optimal' weights :math:`w`, i.e., by the
   terms 'accurately' and 'generalizes' of the introductory discussion, by the
   choice of the corresponding loss function.

This freedom leads to a rich class of linear classifiers, parametrized by the
choice of activation function :math:`\alpha(z)` and the form of loss function
:math:`L(w)`. As there is great freedom in this choice we must understood
better how we can encode certain learning objectives in this choice. We shall
look in more detail at two different choice from :eq:`eq-alpha-z` and
:eq:`eq-L` in the next chapters. One of them results in the
industry-proven 'support vector machine' (SVM) model.

.. container:: toggle
        
    .. container:: header
    
        Homework

    .. container:: homework

        1. Discuss how the 'optimal' choice of weights is influence by changing
           the loss function :eq:`eq-L` to

           .. math:: 
       
               L(w) := \|w\|_p + \frac12 \sum_{i=1}^M \left(y^{(i)}
               -\alpha(w\cdot x^{(i)})\right)^2,

           where :math:`\|w\|_p := (\sum_i |w_i|^p)^{1/p}` is the usual
           :math:`L^p`-norm for :math:`p\in \mathbb N\cup\{\infty\}`.

        2. Give an example of a loss function employing a notion of
           distance other than the Euclidean one and implement the
           corresponding Adaline.


Sigmoid activation function
---------------------------

One of the widely employed choices in machine learning models that are based on
the Adaline framework are sigmoid functions. This class of functions is usually
defined as bouded differentiable function :math:`\mathbb R\to\mathbb R` with a
non-negative derivative. Two frequent examples are the logistic function


.. math::
    \alpha: \mathbb Z \to [0,1], \qquad \alpha(z):=\frac{1}{1+e^{-z}}
    :label: eq-logistic

and the hyperbolic tangent

.. math::
    \alpha: \mathbb Z \to [-1,1], \qquad \alpha(z):=\tanh(z) =
    \frac{e^z-e^{-z}}{e^z+e^{-z}}.

.. container:: toggle
        
    .. container:: header
    
        Homework

    .. container:: homework

        Prove that any sigmoid function has a pair of two horizontal
        asymptodes for :math:`x\to\pm\infty`.

Compared to the linear activation in :eq:`eq-alpha-z` these functions often
have several advantages. 

First, their ranges are bounded by definition. In the cases considered above,
in which there are only two class labels, say or :math:`\{0,1\}` or
:math:`\{-1,+1\}`, the linear activation function :eq:`eq-alpha-z` may attain
values much lower or higher that the numbers used to encode the class labels.
This may lead to extreme updates according to the update rule
:eq:`eq-adaline-update` as the difference :math:`(y^{(i)}-\alpha(w\cdot
x^{(i)}))` may become very large. In the Adaline case considered above in which
we only had one output signal we did not encounter problems during training
because of such potentially large signals. On the contrary they were rather
welcome as they pushed the weight vector strongly in the right direction.
However, soon we will discuss the Adaline for multiple output signals
representing a whole layer of neurons, and furthermore, stack them on top of
each other to form multiple-layer networks of neurons. In these situation such
extreme updates will be undesired and the sigmoid function will work to prevent
them. Note that if we then replace the linear activation by the hyperbolic
tangent above, we ensure :math:`(y^{(i)}-\alpha(w\cdot x^{(i)}))\in
(-2,+2)`, i.e., the updates are uniformly bounded as it was the case for the
Perceptron.

Second, the reason to consider multi-layer networks is to treat general
non-linear classification problems and the necessary non-linearity will enter
into the model by means of such non-linear activation functions. 

Although both reasons are more important for the multi-layer networks
considered later, it will be convenient to discuss the training of sigmoidal
Adaline already for the simply case of only one output signal.  Furthermore,
there is a third reason why, e.g., the logistic is often preferred over the linear
activation as the former can be interpreted as a probability.

The implementation of a sigmoidal Adaline is straight forward. We replace the
activation function :math:`\alpha` accordingly, compute its derivative and adapt
the update rule :eq:`eq-adaline-update`.  

.. container:: toggle
            
    .. container:: header
        
        Homework

    .. container:: homework

        Adapt our Adaline implemetation to employ 

        1. the hyperbolic tangent activation function and
        2. the logistic function   

        and observe the corresponding learning behavior. 

The implementation for the Iris data set should work out of the box having
initial weight set to zero. However, one may recognize that the training for
extreme choices of initial weights will require many epochs of training in
order to achieve a reasonable accuracy. The following plots illustrate the
learning behavior of a linear and hyperbolic tangent Adaline: 

.. plot:: ./figures/python/sigmoid-saturation.py
    :width: 80%
    :align: center

Both plots show the quadratic loss functions :eq:`eq-L` per iteration of
training of a linear (left) and a hyperbolic tangent (right) Adaline. Both
started with an initial weight of :math:`-3` and were presented the signle
training data element :math:`(x^{(1)},y^{(i)})=(1,1)`. The initial weight
was chosen far off a reasonable value. Nevertheless, the linear Adaline
learns to adjust the weight rather quickly while the hyperbolic tangent
Adaline takes about more than two magnitudes more iteration before a
significant learning progress can be observed.

For simplicity and to draw a nice connection to statistics, let us look
at the logistic Adaline model, i.e., the Adaline model with :math:`\alpha(z)`
being the logistic function :eq:`eq-logistic`. The same line of reasoning that
will be developed for the logistic Adaline will apply to the hyperbolic tangent
Adaline in the plot above.
  
Looking at our update rule :eq:`eq-adaline-update` we can read off the
explanation for the slow learning phenomenon. Recall the update rule:
                     
.. math:: w \mapsto w^{\text{new}} := w + \eta \sum_{i=1}^M \left(
   y^{(i)}-\alpha(w\cdot x^{(i)}) \right) \alpha'(w\cdot x^{(i)}) x^{(i)} 

and the derivative of the logistic function :eq:`eq-logistic`:

.. math::

    \alpha'(z) = \frac{e^{-z}}{(1+e^{-z})^2}=\alpha(z)(1-\alpha(z)).

Clearly, for large values of :math:`z` the derivative :math:`\alpha'(z)`
becomes very small, and hence, the update computed by the update rule
:eq:`eq-adaline-update` will be accordingly small even for the case of a
misclassification. This is why this phenomenon is usually referred to as
*vanishig gradient problem*.

If, for whatever reason, we would like to stick with the logistic function as
activation function we can only try to adapt the loss function :math:`L(w)` in
order to better the situation. How can this be done? Let us restrict our
consideration to loss functions of the form

.. math::
   L(w) = \sum_{i=1}^M l(y^{(i)},\alpha(w\cdot x^{(i)}).
   :label: eq-loss-small-l

We compute

.. math::
   \frac{\partial L(w)}{\partial w} 
   = 
   \sum_{i=1}^M \frac{\partial l}{\partial z}(y^{(i)},z)
   \big|_{z=\alpha(w\cdot x^{(i)})}
   \cdot \alpha'(w\cdot x^{(i)}) \, x^{(i)}.
   :label: eq-update-small-l

This means that the only choice to compensate a potential vanishing gradient
due to :math:`\alpha'` is to choose a good function :math:`l`. Bluntly this
could be done by choosing :math:`\frac{\partial l}{\partial z}(y^{(i)},z)
\big|_{z=\alpha(w\cdot x^{(i)})}` to be proportional to the inverse of
:math:`\alpha'(w\cdot x^{(i)})` and then integrating it -- hoping to find a
useful loss function for the training objective. We will not do this but use
this opportunity to motivate a good candidate of the loss function by ideas
drawn from statistics. 

For this we introduce the concept of *entropy* and *cross-entropy*. We
define:

.. container:: definition

    **Definition (Entropy)** Given a discreet probability space
    :math:`(\Omega,P)` we define the so-called entropy by

    .. math::
        H(P) := \sum_{\omega \in \Omega} P(\omega) \, (-1)\, \log_2 P(\omega).

Heuristically speaking, the entropy function :math:`H(P)` measures how many
bits are on average necessary to encode an event. Say Alice and Bob want to
distinguish a number of :math:`N` events but only have a communication channel
through which one bit per communication can be send. An encoding system that is
able distinguish :math:`N` events but on average minimizes the number of
communications between Alice and Bob would allocate small bit sequences for
frequent events and longer ones for seldom events. The frequency of an event
:math:`\omega\in\Omega` is is determined by :math:`P(\omega)` so that the
number of bits necessary to allocate for event :math:`\omega` is given by
:math:`-\log_2(P(\omega))` -- note that :math:`P(\omega)\in[0,1]`.

Let us regard a three
examples:

1. *A fair coin:* The corresponding probability space can be modelled
   by

   .. math::
       \Omega = \{0,1\}, \qquad \text{and} \qquad \forall \omega\in\Omega: \quad
       P(\Omega):=\frac{1}{2}

   so that we find

   .. math::
       H(P) = -\log_2\frac12 = 1.

   Hence, on average we need 1 bit to store the events as typically we have 0
   or 1.

2. *A fair six-sided dice:* The corresponding probability space can be modelled
   by

   .. math::
       \Omega = \{1,2,3,4,5,6\}, \qquad \text{and} \qquad \forall \omega\in\Omega: \quad
       P_\text{fair}(\Omega):=\frac{1}{6}

   and we find

   .. math::
       H(P_\text{fair}) = -\log_2\frac{1}{6} \approx 2.58\ldots.

   Hence, on average we need 3 bits to store which of the six typical events
   occurred.

3. *An unfair six-sided dice:* Let us take again :math:`\Omega=\{1,2,3,4,5,6\}`
   but instead of the uniform distribution like above we chose:

   +---------------------------------+-------------+--------------+--------------+--------------+--------------+-------------+
   | :math:`\omega`                  | :math:`1`   | :math:`2`    | :math:`3`    | :math:`4`    | :math:`5`    | :math:`6`   |
   +---------------------------------+-------------+--------------+--------------+--------------+--------------+-------------+
   | :math:`P_\text{unfair}(\omega)` | :math:`1/4` | :math:`1/16` | :math:`1/16` | :math:`1/16` | :math:`1/16` | :math:`1/2` |
   +---------------------------------+-------------+--------------+--------------+--------------+--------------+-------------+

   In this case we find

   .. math::
       H(P_\text{unfair}) = 2

   Since typically event :math:`\omega=6` occurs more often then the others, on
   average, we need less bits to represent it. In turn, Alice and Bob would we
   need less bits on average for the communication than in the case of the fair
   version of the dice. 

In statistics the true probability measure is usually unknown and the objective
is to find a good estimate of it taking in account the empirical evidence. A
candidate for a measure of how good such a guess is is given by the so-called
*cross-entropy* which we define now.

.. container:: definition

    **Definition (Entropy)** Given a discreet probability space
    :math:`(\Omega,P)` and another measure :math:`Q`  we define the so-called
    cross-entropy by

    .. math::
        H(P,Q) := \sum_{\omega \in \Omega} P(\omega) \, (-1)\, \log_2 Q(\omega).

One may interpret :math:`H(P,Q)` as follows: If :math:`Q` is an estimate of the
true probability measure then :math:`-\log_2 Q(\omega)` is the number of bits
necessary to encode the event :math:`\omega` according to our estimate. The
cross-entropy :math:`H(P,Q)` is therefore an average w.r.t. to the true measure
:math:`P` of the number of bits necessary to encode the events
:math:`\omega\in\Omega` according to :math:`Q`. If according to :math:`Q` we
would allocate the wrong amount of bits to encode the events Alice and Bob
would on average have to exchange more bits per communication. This indicates
that :math:`H(P,P)=H(P)` must be the optimum which is true:

.. container:: theorem

    **Theorem (Cross-Entropy)** Let :math:`(\Omega,P)` be a discreet
    probability space and :math:`Q` another measure on :math:`\Omega`. Then we
    have:

    .. math::
        H(P,Q) \geq H(P,P).

.. container:: toggle
        
    .. container:: header
    
        Homework

    .. container:: homework

       Prove the theorem. *Hint:* Consider first the case of only two possible
       events, i.e., :math:`|\Omega|=2` and find the global minimum.

This property qualifies :math:`H(P,Q)` as a kind of distance between a
potential guess :math:`Q` of the true probability :math:`P`.  After this
excursion to statistics let us employ this distance and with it build a loss
function for the logistic Adaline by the following analogy.

For the logistic Adaline we assume the labels for features :math:`x^{(i)}` to be
of the form :math:`y^{(i)}\in\{0,1\}`, :math:`1\leq i\leq M`. Furthermore, we
observe that by definition also the activation functions evaluate to
:math:`\alpha(w\cdot x^{(i)})\in(0,1)`. This allows to define for the sample space

.. math:: 
    \Omega=\left\{\,\{x^{(i)}=y^{(i)}\}\, \,\big|\,1\leq i\leq M\right\}
    \bigcup\left\{\,\{x^{(i)}=1-y^{(i)}\}\, \,\big|\,1\leq i\leq M\right\}

the following probability distributions 

.. math::
    P(x^{(i)}=y^{(i)}) = \frac{y^{(i)}}{M} 
    \qquad &\text{and} \qquad 
    P(x^{(i)}=1-y^{(i)})=\frac{1-y^{(i)}}{M},\\
    Q(x^{(i)}=1-y^{(i)}) = \frac{\alpha(w\cdot x^{(i)})}{M} 
    \qquad &\text{and} \qquad 
    Q(x^{(i)}=1-y^{(i)})=\frac{1-\alpha(w\cdot x^{(i)})}{M}.


Now we interpret the probability measure :math:`P` which was defined by the
training data as the true measure and :math:`Q` as our estimate of that
measure. The cross-entropy is hence defined as

.. math::
   H(P,Q) &= 
   \sum_{\omega\in\Omega} P(\omega)\log_2 Q(\omega)\\
   &=
   \frac{1}{\log 2}
   \left(
   1-\frac{1}{M} \sum_{i=1}^M 
   \left(
        y^{(i)} 
        \log(\alpha(w\cdot x^{(i)})
        +(1-y^{(i)})
        \log(1-\alpha(w\cdot x^{(i)})
   \right) \right).

Dropping the irrelevant constants we may define a new loss function
:math:`L(w)` by using the following expression for :eq:`eq-loss-small-l`

.. math::
    l(y, \alpha(w\cdot x)):=  
        y
        \log(\alpha(w\cdot x)
        +(1-y)
        \log(1-\alpha(w\cdot x)

so that we get

.. math::
    L(w)= -\frac{1}{M} \sum_{i=1}^M 
    \left(
        y^{(i)} 
        \log(\alpha(w\cdot x^{(i)})
        +(1-y^{(i)})
        \log(1-\alpha(w\cdot x^{(i)})
    \right).

We compute the derivative

.. math::
    \frac{\partial L(w)}{\partial w}
    &=
    -\sum_{i=1}^M
    \left(
        \frac{y^{(i)}}{\alpha(w\cdot x^{(i)})}
        -\frac{1-y^{(i)}}{1-\alpha(w\cdot x^{(i)})}
    \right)
    \alpha'(w\cdot x^{(i)})\, x^{(i)}\\
    &=
    \sum_{i=1}^M
    \left(
        \alpha(w\cdot x^{(i)})-y^{(i)} 
    \right)\,x^{(i)}.

We observe, that the vanishing gradient behavior of :math:`\alpha'` is
compensated by the derivative of the cross-entropy :math:`l'`. In conclusion,
we find the update rule corresponding to this new loss function

.. math:: w \mapsto w^{\text{new}} := w + \eta 
    \sum_{i=1}^M
    \left(
        \alpha(w\cdot x^{(i)})-y^{(i)} x^{(i)}
    \right).

A comparison with the previous update rule :eq:`eq-update-small-l` shows that
with the help of a change of loss function we end up with a update rule that
will not show the vanishing gradient problem. As a rule of thumb one can expect
that logistic Adalines will almost always be easier to train with cross entropy
loss functions unless the vanishing gradient effect is desired -- at a
later point we may come back to this point and discuss that, e.g., in
convolution networks the ReLu activation function (being zero for negative
arguments and linear for positive ones) have actually proven to be very
convenient. For now the take away from this section is that the choices in
:math:`\alpha(z)` and :math:`L(w)` must be carefully tuned w.r.t. each other.

.. container:: toggle
            
    .. container:: header
        
        Homework

    .. container:: homework

        Adapt our Adaline implemetation with the logistic activation function
        and replace the old loss function by the cross-entropy and compare the
        learning behavior in both cases. 


Support Vector Machine
----------------------

* While the Adaline loss function was good a measure of how accurately the
  training data is classified, it did not put a particular emphasis on how the
  optimal weights :math:`w` may generalize for the training data to unseen data;

* Next, we shall specify such a sense and derive a corresponding loss function; 


Linear seperable case
~~~~~~~~~~~~~~~~~~~~~

* Consider a typical linear seperable case of training data. Depending on the
  initial weights both, the Adaline and Perceptron, may find different
  separation hyperplanes of the same training data, however, among all of the possible seperation hyperplanes there is a special one:

    .. figure:: ./figures/keynote/keynote.006.jpeg
        :width: 80%
        :align: center

* The special seperation hyperplane maximizes the margin width of the seperation.

* Note that the minimal distance of a point :math:`x` and the separation
  hyperplane defined by :math:`w` is given by

  .. math::

    \operatorname{dist_w}(x) := \frac{|w\cdot x|}{\|\mathbf w\|};

  recall that :math:`w=(w_0,\mathbf w)`.

* Furthermore, note that the separation of the training data into the classes
  +1 and -1 given by the signum of :math:`w\cdot x^{(i)}` is scale invariant.

.. todo:: 

    Under construction. See for example :cite:`vapnik_statistical_1998`, :cite:`mohri_foundations_2012`.

    * Distance between point and hyperplane in normal form.
    * Scale invariance of :math:`w\cdot x=0`.
    * Minimization problem has a unique solution.


Soft margin case
~~~~~~~~~~~~~~~~

.. todo:: 

    Under construction. See for example :cite:`vapnik_statistical_1998`, :cite:`mohri_foundations_2012`.

    * Minimization problem still has a unique solution.
    * Meaning of slack variables.


