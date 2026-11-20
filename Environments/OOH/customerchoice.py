from math import exp
from numpy.random import gumbel
import numpy as np

class customerchoicemodel(object):
    def __init__(self,
                 base_util,
                 dist_scaler,
                 euclidean,
                 dist_mat,
                 n_cust,
                 beta_k=0.02,
                 outside_util=None):
        self.euclidean_distance = euclidean
        self.dist_scaler = dist_scaler
        self.base_util = base_util
        self.dist_mat = dist_mat
        self.n_cust = n_cust
        self.beta_k = beta_k
        self.outside_util = outside_util
        if len(self.dist_mat)>0:
            self.mnl = self.mnl_distmat
        else:
            self.mnl = self.mnl_euclid

    def ooh_utility(self,distance):
        """paper Eq.(5): u_k = -beta^k * exp[d]; distance divided by dist_scaler to keep exp bounded.
        Exponent capped at 50 to avoid overflow when calibration probes small scalers
        (utility ~ -1e20, effectively minus infinity)."""
        return self.base_util - self.beta_k*exp(min(distance/self.dist_scaler,50.0))

    def mnl_euclid(self,customer,parcelpoint):
        """
        multi-nomial logit model calculating euclidean distance
        """
        distance = self.euclidean_distance(customer.home,parcelpoint.location)#distance from parcelpoint to home
        return self.ooh_utility(distance)

    def mnl_distmat(self,customer,parcelpoint):
        """
        multi-nomial logit model using distance matrix
        """
        distance = self.dist_mat[customer.id_num][parcelpoint.id_num]#distance from parcelpoint to home
        return self.ooh_utility(distance)
    
    def customerchoice_offer(self,customer,action,parcelpoints):
        """
        Customer choice model for the offering decision, i.e., action is 1 parcelpoint offer
        """
        pps = parcelpoints[action-self.n_cust]
        shape = (len(action)+1, 1)
        utils= np.empty(shape)
        utils[0]=self.base_util+customer.home_util
        for idx,pp in enumerate(pps):
            utils[idx+1] = self.mnl(customer,pp)
        utils = np.add(utils,gumbel(0,1, np.shape(utils)))#mu=0,beta=1 (std Gumbel)
        
        idx = np.argmax(utils)
        if idx==0:
            return customer.home, False, -1, 0#home delivery
        else:
            return pps[idx-1].location, True, pps[idx-1].id_num,0#accept offer

    def customerchoice_pricing(self,customer,action,parcelpoints):
        """
        Customer choice model for the pricing decision, i.e., action is vector of prices for all PPs and home delivery
        """
        pps = parcelpoints[parcelpoints.mask].data
        shape = (len(pps)+1, 1)
        utils= np.empty(shape)
        utils[0]=self.base_util+customer.home_util
        for idx,pp in enumerate(pps):
            utils[idx+1] = self.mnl(customer,pp)
        utils = np.add(utils,customer.incentiveSensitivity*action.reshape((len(action),1)))#incentive
        if self.outside_util is not None:#outside option (requirement 2): utility V0, no price term
            utils = np.vstack((utils, [[self.outside_util]]))
        utils = np.add(utils,gumbel(0,1, np.shape(utils)))#mu=0,beta=1 (std Gumbel)

        idx = np.argmax(utils)
        if self.outside_util is not None and idx == len(pps)+1:
            return None, False, -2, 0#customer quits (outside option), -2 sentinel
        if idx==0:
            return customer.home, False, -1, action[0]#home delivery
        else:
            return pps[idx-1].location, True, pps[idx-1].id_num,action[idx]#accept offer

    def customerchoice_home_only(self,customer,action,parcelpoints):
        """
        NoOOH baseline: no OOH offered. Without outside option, always home.
        With outside_util, MNL over {home, quit} (price enters home utility only).
        """
        if self.outside_util is None:
            return customer.home, False, -1, action[0]#pass the home price through
        home_u = (self.base_util + customer.home_util
                  + customer.incentiveSensitivity * float(action[0]))
        utils = np.array([[home_u], [self.outside_util]], dtype=float)
        utils = np.add(utils, gumbel(0, 1, np.shape(utils)))
        if int(np.argmax(utils)) == 1:
            return None, False, -2, 0#quit
        return customer.home, False, -1, action[0]

    def customerchoice_ooh_only(self,customer,action,parcelpoints):
        """
        OnlyOOH baseline: home not in the choice set, MNL over offered OOH points.
        With outside_util, quit is appended so customers may leave instead of booking OOH.
        """
        pps = parcelpoints[parcelpoints.mask].data
        has_capacity = any(pp.remainingCapacity > 0 for pp in pps)
        if not has_capacity and self.outside_util is None:
            raise RuntimeError("OnlyOOH: all offered OOH locations have zero remaining capacity, no feasible choice")
        shape = (len(pps), 1)
        utils = np.empty(shape)
        for idx,pp in enumerate(pps):
            if pp.remainingCapacity > 0:
                utils[idx] = self.mnl(customer,pp)
            else:
                utils[idx] = -1e9#infeasible, never chosen
        if self.outside_util is not None:
            utils = np.vstack((utils, [[self.outside_util]]))
        utils = np.add(utils,gumbel(0,1, np.shape(utils)))#mu=0,beta=1 (std Gumbel)

        idx = int(np.argmax(utils))
        if self.outside_util is not None and idx == len(pps):
            return None, False, -2, 0#quit
        return pps[idx].location, True, pps[idx].id_num, 0