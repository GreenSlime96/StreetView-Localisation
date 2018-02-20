import numpy as np

from scipy.stats import norm

class NaiveFilter:
    def __init__(self, dataset):
        self.dataset = dataset
        self.reset()        

    def predict(self, votes):
        max_votes = np.where(votes == np.amax(votes))[0]

        if len(max_votes) != 1:
            # break tie by finding the vote with most supporters
            # more support = less distance between bits...
            d = self.dataset.distances[max_votes,:][:,max_votes]
            index = max_votes[np.argmin(np.sum(d, axis=0))]
        else:
            # naively pick the most-occuring vote
            index = max_votes[0]

        return index

    def reset(self):
        self.votes = []




class SequenceFilter(Filter):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.reset()

    def reset(self):
        self.states = []

    def predict(self, votes)
        pass


class Viterbi(SequenceFilter):
    def __init__(self, dataset):
        super.__init__(dataset)

        # compute transmission probabilities using the 2nd
        # most distant point as the standard deviation
        sd = np.sort(dataset.distances)[:, 2][:, None]
        self.A = norm.logpdf(dataset.distances, 0, sd)

    def predict(self, votes):
        self.states.append(votes)

        sd = np.sort(self.dataset.distances)[:, 2]
        em = 

        T1 = np.zeros((len(self.states), len(votes)))
        T2 = np.zeros_like(T1)

        T1[:, 0] = self.states[0]
        T2[:, 0] = 0

        for i in range(1, len(self.states)):
            for j in range(len(votes)):
                probs = T1[:, i - 1] * self.A[j]
                T1[j, i] = np.amax()


class Particle(SequenceFilter):
    def __init__(self, dataset):
        super.__init__(dataset)


def viterbi(dataset, votes):
    # smooth the votes by adding one
    votes = votes + 1

    # calculate log-probabilities
    probs = votes / votes.sum(axis=1, keepdims=True)
    B = np.log(probs)

    # maybe replace 0 with dataset.min_dist[:,None]
    sd = np.sort(dataset.distances)[:, 2]
    A = norm.logpdf(dataset.distances, 0, sd[:, None])

    T1 = np.zeros((len(votes), len(votes[0])))
    T2 = np.zeros_like(T1, dtype=np.uint16)

    T1[0] = B[0]
    T2[0] = 0

    mask = np.arange(len(votes[0]))
    for i in range(1, len(votes)):
        temp = T1[i - 1] + A + B[i]
        T2[i] = np.argmax(temp, axis=1)
        T1[i] = temp[mask, T2[i]]

        # c1 = np.argmax(T1[i])
        # c2 = np.argmax(T1[i - 1])

        # if dataset.distances[c1][c2] > 100:
        #     T1[i] = T1[i - 1]
        #     T2[i] = T2[i - 1]

    Z = np.zeros(len(votes), dtype=np.uint16)
    Z[-1] = np.argmax(T1[-1])

    for i in range(len(votes) - 1, 0, -1):
        Z[i - 1] = T2[i, Z[i]]

    return (T1, T2, Z)
