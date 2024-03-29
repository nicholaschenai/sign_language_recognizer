import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_score = float("inf")
        for num_states in range(self.min_n_components,self.max_n_components+1):
            try:
                hmm_model = self.base_model(num_states)
                logL = hmm_model.score(self.X, self.lengths)
                # Note to self: p is not so obvious, refer to the formula found at
                # https://discussions.udacity.com/t/number-of-parameters-bic-calculation/233235/17
                p = num_states ** 2 + 2 * num_states * hmm_model.n_features - 1
                logN = np.log(len(self.X))
                BIC_score = -2 * logL + p * logN
                if BIC_score < best_score:
                    best_model = hmm_model
                    best_score = BIC_score
            except:
                continue
            
        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_model = None
        best_score = float("-inf")
        for num_states in range(self.min_n_components,self.max_n_components+1):
            try:
                hmm_model = self.base_model(num_states)
                logL = hmm_model.score(self.X, self.lengths)
                # vector to store anti-loglikelihoods
                anti_logL_vec = [hmm_model.score(X_other, lengths_other) 
                for word, (X_other, lengths_other) in self.hwords.items() if word != self.this_word]
                DIC_score = logL - np.mean(anti_logL_vec)
                if DIC_score > best_score:
                    best_model = hmm_model
                    best_score = DIC_score
            except:
                continue
            
        return best_model

class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_num_components = None
        best_logL = float("-inf")
        for num_states in range(self.min_n_components,self.max_n_components+1):
            logL_vec = [] # stores log likelihoods for each cv fold
            try:
                if len(self.sequences) > 1:
                    # use 3 fold CV if possible
                    split_method = KFold(n_splits = 2 if len(self.sequences) == 2 else 3)
                    for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                        X_train, lengths_train = combine_sequences(cv_train_idx,self.sequences)
                        hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                            random_state=self.random_state, verbose=False).fit(X_train, lengths_train)
                        X_test, lengths_test = combine_sequences(cv_test_idx,self.sequences)                   
                        logL_vec.append(hmm_model.score(X_test, lengths_test))
                else:
                    hmm_model = self.base_model(num_states)
                    logL_vec.append(hmm_model.score(self.X, self.lengths))
                avg_logL = np.mean(logL_vec)
                if avg_logL > best_logL:
                    best_num_components = num_states
                    best_logL = avg_logL
            except:
                continue
        # return base model trained on the whole training set 
        return self.base_model(best_num_components)
            
                
                                    
            
        
