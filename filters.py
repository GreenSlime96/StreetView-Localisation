import numpy as np

from scipy.stats import norm


class Filter(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.reset()

    def predict(self, votes, infer):
        if not infer:
            self.states.append(votes)

    def get_path(self):
        return [self.predict(v, True) for v in self.states]

    def reset(self):
        self.states = []


class Naive(Filter):
    def __init__(self, dataset):
        super().__init__(dataset)

    def predict(self, votes, infer=False):
        super().predict(votes, infer)

        max_votes = np.where(votes == np.amax(votes))[0]

        if len(max_votes) != 1:
            # break tie by finding the vote with most supporters
            # more support = less distance between bits...
            d = self.dataset.distances[max_votes, :][:, max_votes]
            index = max_votes[np.argmin(np.sum(d, axis=0))]
        else:
            # naively pick the most-occuring vote
            index = max_votes[0]

        return index


class Viterbi(Filter):
    def __init__(self, dataset, epsilon=0.001):
        self.epsilon = epsilon

        # compute transmission probabilities using the 2nd
        # most distant point as the standard deviation
        sd = np.sort(dataset.distances)[:, 2][:, None]
        self.A = norm.logpdf(dataset.distances, 0, sd)

        super().__init__(dataset)

    def predict(self, votes, infer=False):
        super().predict(votes, infer)

        # create new variable and calculate log-probability
        probs = votes + self.epsilon
        probs /= probs.sum()
        B = np.log(probs)

        if not self.T1:
            self.mask = np.arange(len(votes))
            t2 = np.zeros(len(votes), dtype=np.uint16)
            t1 = B
        else:
            temp = self.T1[-1] + self.A + B
            t2 = temp.argmax(axis=1).astype(np.uint16)
            t1 = temp[self.mask, t2]

        self.T1.append(t1)
        self.T2.append(t2)

        return t1.argmax()

    def get_path(self):
        Z = np.zeros(len(self.states), dtype=np.uint16)
        Z[-1] = np.argmax(self.T1[-1])

        for i in range(len(self.states) - 1, 0, -1):
            Z[i - 1] = self.T2[i][Z[i]]

        return Z

    def reset(self):
        super().reset()

        self.T1 = []
        self.T2 = []


class Particle(Filter):
    def __init__(self, dataset, n_particles=10000, velocity=5):
        # particle transitions modelled by constant velocity, normalised to
        # represent the probabilities of transitions from a state
        self.transitions = norm.pdf(dataset.distances, velocity, 2 * velocity)
        self.transitions /= self.transitions.sum(axis=1, keepdims=True)

        self.n_particles = n_particles

        super().__init__(dataset)

    def predict(self, votes, infer=False):
        super().predict(votes, infer)

        # weight and resample
        particles = self.particles * votes
        particles *= self.n_particles / particles.sum()

        # motion update
        self.particles = np.dot(self.transitions, particles)

        return particles.argmax()

    def reset(self):
        super().reset()

        self.particles = np.empty(len(self.dataset.coordinates))
        self.particles.fill(self.n_particles / len(self.particles))

    def get_path(self):
        particles = self.particles
        self.particles.fill(self.n_particles / len(particles))

        path = super().get_path()

        self.particles = particles

        return path
