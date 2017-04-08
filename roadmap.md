# Roadmap: MAML Lecture Notes

## Organizational Points

### Basic Principles:

1. Small chapters: Ideally 1 point per chapter;
1. Clear distinction between informal and formal parts;
1. Small and self-contained source code per application;
1. Graph or map of the book chapter and sections;
1. Guest authors from the community.

### Open questions

* Which data sets can we use for as training data? 
* We need to distinguish between linear separable and linear non-separable
data.
* Which are good and accessible source for the universal approximation theorems?

### Cherry picking in the existing literature

* Good reference for the Python level that we should assume: 

    * Raschka: "ML with python"

        Only first 1-2 chapters.
    
* Good reference for important points to make:

    * Michael Nielson: "Deep Learning"

        However, explanation to lengthy for our intended audience.    

* "Mathematical Foundations of Machine Learning"

## Skeleton

1. Introduction
    * Who is the reader?
        Bridge btw. math/phys. jargon and computer science for ML.
        At least calculus/analysis I.
    * What is ML?
        * The dream of AI
        * Pragmatic intermediate goal for ML
        * History
            * AI Winters
            * Why again interesting?
                * Most algorithms are from the 70ies
                * Big data available
                * Computer resources
        * Perspective: modern ML
    * Refs of helpful books
    * ML vocabulary 
        * Supervised
        * Reinforced
        * Unsupervised
    * What will we do? Why another book?
        * Aim
        * Theory (formal)
        * Implementation
        * Challenges heuristics, Learning strategies, Data preparations
        (informal)

    More in the vain of given a perspective of the book towards modern ML.
    Not pretending that we are the experts. Where we are, what we want to do.

1. Linear classification
    * Perceptron
        * Learning rule
            How math. encode a mode of learning in a model.
        * How to prove learn ability?
        * Python implementation
            
    * Adaline
        * Gradient descent
        * Batch learning
        * Learning parameter
        
        New idea: use of analytic optimization techniques.
        Used everywhere.

    * Learning the unknown
        * Two goals: 
            * Classification of training data
            * Classification of unknown data
    
    * Support vector machine
        * Notion how to generalize to unknown data
            * Hard-margin case
            * Soft-margin case
            
1. Non-liear classification
    * Mapping to higher dimensional space
        * Extension of Adaline
        * Extension of SVM
    
    Use the old code (which then must already be general enough to cope with
    arbitrary many feature dims) and only make a script that generates higher
    dim. training data.
            
1. Optimization theory
    * Existence of optimal solutions
    * Proof that SVMs have a unique optimal solution.
    * Convex optimization
    * Optimal conditions for constraint convex programs
    * Derive KKT conditions from hyperplane separation theorems
        * Apply KKT to example constraint optimization program
        * Apply KKT to SVM: Meaning of the support vectors
    * Lagrange function
        * Saddle points
        * Relation between saddle point of L and KKT conditions
        * Primal and dual form
    
    Communicate what gradient descent is about. How constraints complicate the
    picture and what the dual formulation is.

1. SVM in dual form
    * Kernel trick, why better performance
    * relate that back to the map to higher dims.

1. Neural networks
    * Multi-layer networks
    * Learning and update rule: relate that back to Adaline
    * Iris flower classification
    * An efficient training algorithm: Backpropagation

1. Project: Handwritten number classification
    * Implementation
    * Learning behavior and strategies
    * Preparation of Training data etc.
    
1. Representation and approximation by neural networks
    * Representation of boolean functions
    * Representation of binary classification
    * Approximation of real valued functions

1. Outlook
    * Deep learning
    * Recurrent networks
    
1. References
