import numpy as np
import Src.Utils.Utils as Utils
from Src.parser import Parser
from Src.config import Config
from time import time

class Solver:
    def __init__(self, config):
        # Initialize the required variables

        self.config = config
        self.env = self.config.env#env used for training
        self.test_env = self.config.test_env#seperate env used for testing only
        self.state_dim = np.shape(self.env.reset())[0]
        
        #to ensure we do not exceed the fleet capacity
        self.max_steps = int(config.n_vehicles*config.veh_capacity)-1
        
        print("State space: {}".format(self.state_dim))

        self.model = config.algo(config=config)

    # Main training/simulation loop
    def train(self):
        # Learn the model on the environment
        rewards = []

        checkpoint = self.config.save_after
        start_ep = 0

        t0 = time()
        for episode in range(start_ep, self.config.max_episodes):
            # Reset both environment and model before a new episode
            state = self.env.reset()
            self.model.reset()

            step = 0
            done = False

            while not done:
                action = self.model.get_action(state, training=True)
                new_state, done, stats,route_data = self.env.step(action=action)
                state = new_state
                step += 1
                _ = self.model.update(route_data,state,False)
                if step >= self.max_steps or done:
                    travel_time = self.model.update(route_data,state,True)#do full update when episode is done
                    rewards.append(Utils.total_costs(stats[1],stats[2],travel_time,stats[3],stats[6],self.config))
                    break
            
            if episode%checkpoint == 0 or episode == self.config.max_episodes-1:
                print('time required for '+str(checkpoint)+' episodes :' +str(time()-t0))
                Utils.plot_training_curves(rewards,self.config)
                #Utils.save_plots_stats(run_stats,travel_time,run_time,actions=actions,config=self.config,episode=episode)

                t0 = time()

        self.model.save()  # persist final weights when config.save_model is set

    def eval(self, episodes=1):
         
        # Evaluate the model, see run_ppo - eval() for some interesting statistics to save,
        # we removed these statistics tracking from run.py to make the code a bit more readable,
        # but you can add those statistics tracking easily to this file again
         total_cost = []
         actions = []
         accepted_price = []
         price_time=[]
         home_delivery_loc = []
         step_time = []
         for episode in range(episodes):
             state = self.test_env.reset()
             step = 0
             done = False
             while not done:
                 t1 = time()
                 action = self.model.get_action(state, training=False)
                 new_state, done, stats,route_data = self.test_env.step(action=action)
                 price_time.append([stats[7],step])
                 actions.append([*action,step,episode])
                 home_delivery_loc.append([stats[5],step,episode])
                 state = new_state
                 step += 1
                 step_time.append(time()-t1)
                 if step >= self.max_steps:
                     break
         
         #directly save statistics
         #Utils.save_eval_stats(travel_time,total_cost,actions,accepted_price,count_home_delivery,service_time,
                  #              parcel_lockers_remaining_capacity,home_delivery_loc,step_time,price_time,self.config)
             
         return total_cost, accepted_price,step_time

    def eval2(self, episodes=1):
            # Evaluate the model
            # Todo: cleanup, looks a bit messy
            travel_time = []
            total_cost = []
            actions = []
            accepted_price = []
            accepted_discount = []
            count_home_delivery = []
            count_quit = []
            service_time = []
            parcel_lockers_remaining_capacity = []
            home_delivery_loc = []
            step_time = []
            for episode in range(episodes):
                # print('episode' ,episode)
                state = self.test_env.reset()
                step = 0
                done = False
                while not done:
                    t1 = time()
                    action = self.model.get_action(state, training=True)

                    new_state, done, stats, route_data = self.test_env.step(action=action)
                    actions.append([*action, step, episode])
                    # accepted_price.append([stats[3],step,episode])
                    home_delivery_loc.append([stats[5], step, episode])
                    state = new_state
                    step += 1
                    step_time.append(time() - t1)
                    if step >= self.max_steps:
                        break
                travel_time.append([self.test_env.reopt_for_eval(route_data), episode])  # short HGS (re-opt) call
                # total_cost.append([Utils.total_costs(stats[1],stats[2],travel_time,stats[3],self.config)[0][0][0],episode])
                service_time.append([stats[2], episode])
                count_home_delivery.append([stats[1], episode])
                accepted_price.append(stats[3])
                accepted_discount.append(stats[6])
                count_quit.append(stats[8])
                for i in stats[4]:
                    parcel_lockers_remaining_capacity.append([i.remainingCapacity, i.location.x, i.location.y, episode])

            cnt = 0
            trvl = 0
            trvl_list = []
            srvc = 0
            fail = 0
            distance = 0
            count_pp = 0
            for i in home_delivery_loc:
                if i[0] > 0:
                    count_pp += 1
                distance += i[0]

            quits = sum(count_quit)

            # 工资 +
            cost_multiplier = (self.config.driver_wage + self.config.fuel_cost * self.config.truck_speed) / 3600
            for i in range(0, len(count_home_delivery)):
                cnt += count_home_delivery[i][0]
                trvl += (travel_time[i][0] * cost_multiplier)
                trvl_list.append(travel_time[i][0] * cost_multiplier)
                srvc += (service_time[i][0] * cost_multiplier)
                fail += count_home_delivery[i][
                            0] * self.config.home_failure * self.config.failure_cost  # costs of failed delivery

            d_list = np.concatenate(accepted_discount)
            r_list = np.concatenate(accepted_price)
            metrics = {
                'pct_home': cnt / len(home_delivery_loc),
                'quit_rate': quits / len(home_delivery_loc),
                'travel_costs': trvl / episodes,
                'service_costs': srvc / episodes,
                'failure_costs': fail / episodes,
                'avg_charge': float(np.mean(r_list)) if len(r_list) else 0.0,
                'std_charge': float(np.std(r_list)) if len(r_list) else 0.0,
                'avg_discount': float(-np.mean(d_list)) if len(d_list) else 0.0,
                'std_discount': float(np.std(d_list)) if len(d_list) else 0.0,
                'charge_revenue': float(np.sum(r_list)) / episodes,
                'discount_costs': float(-np.sum(d_list)) / episodes,
            }
            metrics['total_costs'] = (metrics['travel_costs'] + metrics['service_costs']
                                      + metrics['failure_costs'] + metrics['discount_costs']
                                      - metrics['charge_revenue'])
            served = len(home_delivery_loc) - quits
            metrics['net_profit'] = (self.config.revenue * served / episodes
                                     + metrics['charge_revenue'] - metrics['discount_costs']
                                     - metrics['travel_costs'] - metrics['service_costs']
                                     - metrics['failure_costs'])
            print('percentage home delivery: ', metrics['pct_home'])
            print('travel costs: ', metrics['travel_costs'])
            print('service costs: ', metrics['service_costs'])
            print('failure costs: ', metrics['failure_costs'])
            print('Avg. Charge: ', metrics['avg_charge'], 'std.: ', metrics['std_charge'])
            print('Avg. Discount: ', metrics['avg_discount'], 'std.: ', metrics['std_discount'])
            print('Charge revenue: ', metrics['charge_revenue'])
            print('Discount costs: ', metrics['discount_costs'])
            print('total costs: ', metrics['total_costs'])
            print('quit rate: ', metrics['quit_rate'])
            print('net profit: ', metrics['net_profit'])
            if count_pp > 0:
                print('average travelled by customers: ', distance / count_pp)

            return metrics

def main(train=True):
    t = time()
    args = Parser().get_parser().parse_args()

    config = Config(args)
    solver = Solver(config=config)

    if train:
        solver.train()
    
    #evaluate model
    metrics = solver.eval2(20)
    #Utils.plot_test_boxplot(rewards,prices,step_time,config)
    
    print('total timing: ', time()-t)

if __name__== "__main__":
        main(train=True)

