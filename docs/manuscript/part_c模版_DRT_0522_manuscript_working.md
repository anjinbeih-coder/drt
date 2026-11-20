# Dynamic meeting-point recommendation and pricing for many-to-one demand-responsive transit: A predict-then-optimize approach

Jin Ana, Tian-Liang Liua,1, Guanghui Fua

aMoE Key Laboratory of Complex System Analysis and Management Decision, School of Economics and Management, Beihang University, Beijing 100191, China

# Abstract

Demand-responsive transit (DRT) can complement fixed-route public transport, but door-to-door many-to-one services often create detours, high operating costs, and unreliable schedules. Meeting points can reduce routing burden, yet they introduce a passenger-acceptance tradeoff because users compare walking effort, in-vehicle time, price, and the outside option. This paper studies an online many-to-one DRT setting in which passengers arrive sequentially during the booking window and the operator must quote home-pickup and meeting-point options before the final route is known. We develop Dynamic Meeting-Point Recommendation and Pricing Optimization (DRPO), a DRT-specific predict-then-optimize framework. DRPO follows a `recommend -> predict -> price` pipeline: it recommends feasible boarding options, predicts option-level marginal routing impacts and service-time proxies, and converts these predictions into option-specific price adjustments under passenger choice. The problem is formulated as a Markov decision process, and the prediction component is trained with a blended Huber and SPO+ loss to align cost estimates with downstream pricing. We evaluate DRPO against five strategies on a synthetic benchmark and a semi-realistic Beijing commuter-corridor case. In the synthetic benchmark, DRPO reports $1 2 . 1 4 \%$ lower average total cost than Static-pricing and $6 . 9 4 \%$ higher average net profit than DSPO. In the Beijing case, DRPO reports $1 . 3 5 \%$ lower average total cost and $1 . 8 6 \%$ higher average net profit than DSPO, while reducing abandonment from $4 . 0 4 \%$ to $3 . 2 4 \%$ . These results support integrated meeting-point recommendation, cost prediction, and pricing for many-to-one DRT while tracking passenger participation.

Keywords: demand-responsive transit; many-to-one DRT; meeting point recommendation; dynamic pricing; smart predict-then-optimize (SPO)

# 1. Introduction

Demand-responsive transit (DRT) occupies an important position between fixed-route public transport and private car travel. By adapting routes or stops to realized demand, DRT can improve accessibility in areas where travel demand is dispersed, time-varying, or insufficient to support high-frequency fixed-route service. A common operational pattern is many-to-one service, such as commuters traveling from suburban residential areas to a central employment district or students traveling from dispersed homes to a school. In these settings, the operator must balance service convenience for passengers against the routing cost and schedule reliability of collecting many dispersed requests.

Many-to-one DRT operators can provide home pickup, in which the vehicle visits each passenger's origin, or meeting-point pickup, in which passengers walk to recommended boarding locations. Home pickup is attractive to passengers but can create longer routes, additional stops, and less reliable arrival times. Meeting points reduce detours and simplify vehicle operations, but they shift part of the service burden to passengers and may increase cancellation or non-participation if walking effort or perceived service quality is unacceptable. Prior studies show that meeting points can improve system efficiency when walking requirements remain reasonable (Zheng et al., 2019; Zhang et al., 2023). A key unresolved question is how operators should manage this tradeoff when passengers are allowed to choose between home pickup, recommended meeting points, and not using the service.

This question is dynamic because each accepted boarding decision changes both the current passenger's experience and the future routing burden faced by the operator. Evidence from on-demand transportation and related service systems suggests that passenger acceptance depends on the joint attributes of the offered option, including price, walking distance, booking information, and travel time (Bai et al., 2019; Yan et al., 2025). In practice, however, these attributes must be quoted at the time of booking, while the final route is not known until later requests have arrived. The operator therefore faces a sequential demand-management problem: it must recommend feasible boarding options, anticipate their marginal downstream routing impacts, and set prices that steer choices without causing excessive passenger abandonment.

To address this problem, this paper proposes Dynamic Meeting-Point Recommendation and Pricing Optimization (DRPO) for many-to-one DRT. DRPO is a DRT-specific predict-then-optimize framework rather than a general new pricing theory. It preserves a `recommend -> predict -> price` operational sequence: first, it constructs a feasible menu consisting of home pickup and nearby meeting-point options; second, it predicts option-level marginal insertion-cost proxies and service-time information before the final route is known; third, it converts these predictions into option-specific price adjustments under a passenger choice model. Because prediction errors matter through the prices and choices they induce, the learning component uses an SPO/SPO+ decision-focused objective to align prediction with downstream pricing quality.

The contributions of this paper are fourfold. First, it formulates a many-to-one DRT demand-management problem in which passengers arrive sequentially and choose among home pickup, recommended meeting points, and an outside option after observing option-specific prices and service attributes. Second, it develops DRPO as a dynamic meeting-point recommendation and pricing framework that links feasible option generation, option-level cost and service prediction, and price optimization under passenger choice. Third, it adapts decision-focused predict-then-optimize learning to train marginal-cost predictions for the downstream pricing decision rather than only for pointwise prediction accuracy. Fourth, it evaluates DRPO on a controlled synthetic benchmark and a semi-realistic Beijing commuter-corridor case, comparing operator-side outcomes with passenger acceptance and quit behavior.

The remainder of the paper is organized as follows. Section 2 reviews DRT operations, meeting-point coordination, dynamic pricing, passenger behavior, and decision-focused learning, and then synthesizes the research gap. Section 3 defines the many-to-one DRT decision setting and passenger choice model. Section 4 presents the DRPO framework and its `recommend -> predict -> price` implementation. Section 5 reports the synthetic and semi-realistic experiments, and Section 6 concludes with implications and future research directions.

# 2. Related work

This section positions the study at the intersection of DRT operations, meeting-point coordination, dynamic pricing, passenger choice, and decision-focused learning. Section 2.1 reviews DRT and flexible-transit operations. Section 2.2 discusses meeting-point coordination and pricing as demand-management mechanisms. Section 2.3 reviews passenger behavior and predict-then-optimize learning. Section 2.4 synthesizes the research gap addressed by DRPO.

# 2.1. Demand-responsive transit systems

Recent DRT research has examined system design, operational optimization, and service equity. For system design, Baier et al. (2024) establish a four-step evaluation framework to systematically measure DRT performance. Sorensen et al. (2021) study fully flexible door-to-door DRT services in rural Germany and find that such services tend to concentrate on main traffic routes while remaining inefficient in low-demand areas. Currie and Fournier (2020) review 40 years of DRT implementation and report that the failure rate is high ( $50 \%$ within seven years), mainly because of high operating costs and the complexity of many-to-many service models.

For operational optimization, data-driven and algorithmic methods are widely used to improve the efficiency and adaptability of DRT. Lyu et al. (2019) and Xia et al. (2022) develop route-planning frameworks for customized bus systems that combine demand clustering, dynamic planning, and differential evolution, improving operational efficiency by up to $47 \%$ . Wu et al. (2024) introduce a multi-agent deep reinforcement learning framework for real-time route control in customized bus networks, reducing operating costs without sacrificing service quality. Kostic et al. (2021) propose a two-step deep survival modeling method for electric carsharing systems that combines neural networks with copula-based spatial correlation modeling to improve idle-time prediction and vehicle scheduling. Lee et al. (2021) develop a two-stage stochastic programming model for regional flexible public transport systems, optimizing vehicle routing while maintaining service reliability under demand uncertainty. Liu et al. (2023) use machine learning and spatial analysis to study passenger volumes in Shanghai's customized public transport system, showing that customized public transport can complement metro services in suburban areas.

Equity and service quality have become increasingly prominent in recent DRT research. Liezenga et al. (2024) show that suburban microtransit systems can particularly benefit vulnerable populations, with low-income users accounting for $5 5 . 5 \%$ of passengers, thereby improving travel convenience in underserved areas. Miller et al. (2016) propose the Public Transit Sustainability Mobility Analysis Tool (PTSMAT), which combines environmental, economic, social, and system-efficiency indicators to evaluate public-transport sustainability. Shang et al. (2022) use smart-card data from Beijing and find that, although customized bus systems can alleviate congestion and improve commuting comfort, higher fares may limit accessibility for low-income passengers. Knierim and Schlüter (2021) show that DRT adoption in rural Germany is negatively correlated with car ownership and positively correlated with physical disability and age. These studies highlight the role of DRT in mitigating travel inequality for vulnerable groups.

Incentive strategies and pricing models are also gaining attention for improving DRT performance. Wang et al. (2021) develop a bi-level optimization framework that combines passenger behavioral modeling with vehicle route planning. They found that targeted monetary incentives could reduce detours by $23 \%$ and increase operators' profits by $1 8 \%$ . Li et al. (2021) employ a game-theoretic approach to optimize pricing for customized bus and ride-sharing services. They show that strategic price differentiation based on value of time and vehicle ownership can enhance platform profitability. Liu and Ceder (2015) point out that policy coordination and system integration remain key challenges for the large-scale implementation of customized bus services in China. Although a few studies have examined how pricing can improve operator efficiency in DRT systems, most of them do not fully consider the sequential arrival of passengers and the spatiotemporal distribution of demand.

Recent research on DRT and on-demand ride-sharing shows that passenger flexibility in time and space can improve system efficiency. Wu et al. (2025) and Zhang et al. (2025) propose operating strategies such as dynamic stop assignment and joint supply-demand management to reduce the cost and detours associated with door-to-door DRT, especially in feeder services to rail transit. Fielbaum et al. (2021) and Li et al. (2025) further show that encouraging passengers to walk to optimized meeting points or allowing flexible matching delays can improve operational efficiency and user satisfaction. Overall, these studies indicate that assignment, pricing, and flexible user behavior can jointly support more adaptive and efficient DRT systems.

# 2.2. Meeting points, walking coordination, and pricing

The literature demonstrates advanced approaches to dynamic pricing across industries. Galiullina et al. (2024) and Akkerman et al. (2024) propose stochastic optimization frameworks integrating machine learning with two-stage stochastic programming for last-mile dynamic pricing. In retail supply chains, Kayikci et al. (2022) propose an IoT-driven dynamic pricing model that adjusts food prices in real-time based on freshness indicators, reducing perishable waste by $4 7 . 6 \%$ compared to fixed pricing strategies, while Dong et al. (2009) and Kouvelis et al. (2008) derive optimal pricing policies for substitutable products under inventory constraints. For service industries, Yang et al. (2013) found that the pricing strategy combined with the route planning can increase the profit of fresh e-commerce by $3 . 8 \%$ . The literature shows a key finding: dynamic pricing can increase revenue in many industries such as transportation and e-commerce through price adjustment based on demand.

Meeting-point-based operations have received growing attention because they offer a way to improve routing efficiency without abandoning service flexibility altogether. Early work by Stiglic et al. (2015) showed that introducing meeting points into ride-sharing systems can substantially reduce vehicle mileage while keeping walking distances manageable. Zheng et al. (2019) show that introducing meeting points into flex-route transit can generate sizable operational benefits, and Fielbaum et al. (2021) demonstrate

that optimized walking locations can improve on-demand ridesharing performance. More recent work reaches similar conclusions in richer on-demand settings. Wu et al. (2025) study dynamic DRT scheduling with time-dependent travel times, Zhang et al. (2025) examine fixed-route and demand-responsive transit design with dynamic stops, Li et al. (2025) analyze delayed matching and walking in on-demand ridesharing, Pellegrini and Fielbaum (2025) study users' willingness to accept flexible walking, and Yan et al. (2025) investigate the adoption trade-off between dynamic and static walking. Together, these studies show that flexible stop assignment and walking-based coordination can improve efficiency in on-demand systems. Even so, most studies treat meeting-point assignment as a planning or control problem and pay limited attention to how real-time prices can be used to influence passenger acceptance of meeting points relative to home pickup.

Dynamic pricing has long been studied as a mechanism for balancing demand and operational costs under uncertainty. Gallego and van Ryzin (1994) develop a classical inventory-based model showing how prices can be adapted over finite horizons under stochastic demand. In logistics and delivery settings, Galiullina et al. (2024) and Akkerman et al. (2025) show that dynamic selection and pricing can steer demand toward more efficient service options. Yang et al. (2016) demonstrate that pricing and routing decisions can be profitably coordinated in e-fulfillment, while Dong et al. (2009) characterize optimal pricing under substitution effects. Beyond transportation, Kayikci et al. (2022) propose a data-driven dynamic pricing strategy for reducing perishable food waste, Chen et al. (2017) study dynamic price-vector formation for demand response in photovoltaic-assisted electric-vehicle charging stations, and Zhao and Lee (2022) develop a deep-reinforcement-learning approach for dynamic pricing at electric-vehicle charging stations. In DRT and adjacent mobility services, pricing has also been identified as a promising lever. Wang et al. (2021) show that passenger incentives can reduce detours and improve operator profit, and Li et al. (2021) study strategic pricing in customized bus and ride-sharing settings. Taken together, the literature shows that dynamic pricing can improve revenue or operating performance in several industries by adjusting prices to demand, costs, and service conditions. Yet most of this work remains static or offline in nature, and it rarely models the sequential arrival of passengers together with the need to quote service attributes, such as in-vehicle travel time, before the downstream routing problem is fully realized.

# 2.3. Passenger behavior and decision-focused learning

Equity and user experience remain important concerns in the DRT literature. Liezenga et al. (2024) evaluate access equity in suburban microtransit, Shang et al. (2022) examine the integration of conventional and customized bus services in Beijing, and Knierim and Schlueter (2021) study attitudes toward demand-responsive transport among potentially less mobile people in rural Germany. These access-equity evaluations and empirical studies of customized bus services show that service design affects who benefits from flexible transit and how willing different user groups are to adopt it. These findings highlight the importance of explicitly modeling behavioral responses when designing demand-steering mechanisms in DRT.

The methodological challenge of the present problem is not only to price options online, but also to learn cost predictions that are useful for the downstream pricing decision. This connects the problem to decision-focused learning. Elmachtoub and Grigas (2022) show that prediction models should ideally be trained with respect to the induced downstream optimization loss rather than a purely pointwise prediction loss. Their Smart Predict-Then-Optimize framework provides exactly this perspective. Donti et al. (2017) extend decision-focused learning to task-based end-to-end model learning, and Wilder et al. (2019) develop decision-focused learning methods for combinatorial optimization settings. In platform-based mobility, Yang et al. (2025) apply an SPO-style decision-focused learning framework to ride-hailing subsidy allocation. Mandi et al. (2024) provide a broader survey of decision-focused learning methods and benchmarks. In the present setting, the relevant downstream problem is a pricing problem defined over predicted marginal insertion costs. The paper therefore extends the logic of decision-focused learning to a many-to-one DRT context with meeting-point candidate generation and endogenous passenger choice.

# 2.4. Research gaps and contributions

The literature motivates three gaps addressed in this paper. First, meeting-point design and demand steering have often been studied separately, leaving limited evidence on settings where passengers choose between home pickup and recommended meeting points in response to real-time option-specific prices. Second, much DRT optimization work remains static, offline, or primarily routing-centric, whereas many-to-one services require decisions to be made sequentially as passengers arrive during the booking window. Third, predictive components are commonly trained for pointwise accuracy, even though the operational value of a cost prediction in this setting depends on the pricing decision and passenger choice that the prediction induces.

DRPO addresses these gaps as an integrated predict-then-optimize framework for many-to-one DRT meeting-point recommendation and pricing. The operator first recommends a feasible menu of home-pickup and meeting-point options, then predicts option-level marginal routing impacts and service-time proxies, and finally prices the options under a passenger choice model that includes the outside option. The decision-focused learning component aligns the prediction module with the downstream pricing problem. In this way, DRPO connects behavioral response, dynamic option pricing, and routing-cost prediction while remaining specific to the many-to-one DRT service context.

# 3. Model

This paper studies a many-to-one demand-responsive transit service in which passengers travel from dispersed origins to a common destination and submit requests sequentially during the booking horizon. For each request, the operator prices a menu of service options, including home pickup and recommended meeting points, before the full demand realization and final routing plan are known.

# 3.1. Problem description

In this setting, passenger demand is uncertain and revealed online. When a passenger enters the reservation system, the operator observes the home location and offers a small menu of feasible boarding options. The menu includes the passenger's home location and a set of recommended meeting points. Each option is associated with a fare, walking distance, and predicted in-vehicle travel time. The passenger then chooses one of the offered options or selects the outside option.

The operator uses option-specific price adjustments to influence passenger choices, offering discounts for meeting-point pickup and surcharges for home pickup. This strategy is intended to balance passenger acceptance against the routing burden generated by multiple individual pickups. Fig. 1 contrasts the two boarding modes considered in the paper.

![](images/b6337f2de9a203cac6b7bbf5386966e34dc55ace832122b83886e5a1070ac941.jpg)  
(a) Home pickup

![](images/aa3b0fefd2e0a404cb2d186730c629f085cac1353ab179ddee6986dd9a459840.jpg)  
(b) Walking to recommended meeting points

Fig. 1. Home pickup vs. walking to recommended meeting points in a many-to-one DRT system.

Since the final vehicle routes are determined only after all bookings are finalized, the system must estimate the downstream implication of each current boarding choice under demand uncertainty. Fig. 2 shows the passenger interface in our DRT system.

![](images/15dbec3398a29790659de25940a9e7e0b71bc38823a4f5f9a2dd92dd17a53399.jpg)  
Fig. 2. Passenger interface.

# 3.2. Problem formulation

# 3.2.1. Decision timeline

We simulate the booking process as a series of distinct decision periods within a defined booking interval, $[ 0 , 1 , \dots , T ]$ , where $T$ represents the deadline for bookings. Once the time $T$ has elapsed, no more requests are taken; the operator then addresses the multi-vehicle routing using the actual bookings made during that period. Passenger requests arrive sequentially at epochs $t = 1 , \dots , T$ , with origins drawn from the service-area distribution. At each epoch $t$ , a passenger arrives with probability $\lambda _ { t } \in [ 0 , 1 ]$ and no passenger arrives with probability $1 - \lambda _ { t }$ .

At each epoch, the online interaction follows the sequence shown in Fig. 3.

1. A passenger request arrives at epoch $t$

2. The system recommends several boarding options (including home pickup and meeting points), each with an associated price, walking distance and in-vehicle travel time.   
3. The passenger selects an option $k \in O _ { t }$ or rejects the service.   
4. After the cutoff time $T$ , the operator performs multi-vehicle routing optimization.

Passenger accesses system at time t.   
System shows boarding options.

Homet

M-P1 1 fare:12.5￥ i-v-t:32 min walk:300m

M-P², t fare:9￥ i-v-t: 26 min walk:500 m

P,selects $\mathrm { M - P _ { \mathrm { ~ t ~ } } ^ { 2 } }$

M-P2t t √ selected

![](images/86b8033a97e5f7b44291626d4a7304d55ca942b08a601d16249ad9a82ad39f9d.jpg)  
Fig. 3. Online DRT system workflow.

For passenger $t$ , the booking system offers home pickup $h _ { t }$ and a set of recommended meeting points $M _ { t }$ , which together form the complete range of choices $O _ { t } = \{ h _ { t } \} \cup M _ { t }$ . For example, if ${ { M } _ { t } } \mathrm { ~ = ~ }$ $\{ m p _ { 1 } , m p _ { 2 } , m p _ { 3 } \}$ , then $O _ { t } = \{ m p _ { 1 } , m p _ { 2 } , m p _ { 3 } , h _ { t } \}$ . In this work, $M _ { t }$ is determined heuristically by choosing the $n$ nearest meeting points to the passengers' homes. The home pickup option is always part of the deal, ensuring that passengers can still opt for home pickup.

# 3.2.2. Net profit function

The operator aims to boost its earnings by redirecting some passengers from home pickup to designated meeting points, thus cutting down on transportation expenses without excessive compromising the quality of service. The operating cost contains two components: i) labor cost associated with vehicle operating time, denoted by $c _ { \omega }$ ; ii) fuel cost associated with travel distance, denoted by $c _ { f }$ ; Here, the finalized route plan $R ^ { T }$ is defined to include the final leg from the last served location to the common destination.

Therefore, this final leg is already incorporated into the route-arc set ${ \mathcal { E } } ( R ^ { T } )$ , and its associated costs are accounted for in Eq. (2).

Passenger requests arrive sequentially, and the total number of arrivals follows an unknown demand process $D$ . To capture behavioral heterogeneity, we divide passengers into segments $g \in { \mathcal { G } }$ , where $\mu _ { g }$ denotes the probability that an arriving passenger belongs to segment $g$ .

Once the booking deadline hits, the finalized bookings and route plan become clear. Let $B _ { T }$ denote the accepted booking set at the close of the booking period, and $R ^ { T }$ denote the associated route plan. The operator’s objective is to maximize total net profit:

$$
\max  \Pi \left(B _ {T}\right), \tag {1}
$$

The downstream cost component of $\Pi ( B _ { T } )$ is denoted $C _ { T } ( B _ { T } )$ , so that $\Pi ( B _ { T } ) = \mathrm { R e v e n u e } ( B _ { T } ) -$ $( B _ { T } ) -$ $C _ { T } ( B _ { T } )$ , where R $\begin{array} { r } { ( B _ { T } ) = \sum _ { ( t , k ) \in B _ { T } } ( f _ { e } + a _ { t k } ) } \end{array}$ and

$$
C _ {T} (B _ {T}) = c _ {\omega} \left(\sum_ {(i, j) \in \mathcal {E} (R ^ {T})} \omega_ {i, j} + \sum_ {i \in \mathcal {N} (R ^ {T})} l _ {i}\right) + c _ {f} \sum_ {(i, j) \in \mathcal {E} (R ^ {T})} d _ {i, j}. \tag {2}
$$

In R $( B _ { T } ) , f _ { e }$ is the base fare and $a _ { t k }$ is the option-specific price adjustment. $a _ { t k } < 0$ represents a discount, whereas $a _ { t k } > 0$ represents a surcharge.

In Eq. (2), the first term represents the time-related operating cost, computed as the unit time cost $c _ { \omega }$ multiplied by the total vehicle operating time. Specifically, $\begin{array} { r } { \sum _ { ( i , j ) \in \mathcal { E } ( R ^ { T } ) } \omega _ { i , j } } \end{array}$ is the total travel time over all route arcs, and $\textstyle \sum _ { i \in { \mathcal { N } } ( R ^ { T } ) } l _ { i }$ is the total dwell time at visited nodes. The second term encompasses the operating cost tied to distance, calculated by multiplying the unit cost per distance, denoted as $c _ { f }$ , with the total distance traversed across all route segments. Since $R ^ { T }$ includes the final leg from the last served location to the common destination, the relevant time and distance cost are already included in the second and third components.

# 3.3. Dynamic programming modeling and the Bellman equation

The online booking problem is inherently dynamic. Before the deadline $T$ , the operator does not know how many additional passengers will arrive or how future requests will affect the final vehicle routes. A boarding-location decision for the current passenger changes the accepted booking set and therefore changes the downstream routing problem faced at the end of the booking horizon. We model this sequential interaction as a finite-horizon Markov decision process (MDP).

Let $V _ { t } ( s _ { t } )$ denote the maximum expected net profit from the beginning of epoch $t$ to the end of the booking period, given system state $s _ { t }$ . Because route operating costs are realized after the booking horizon is complete, the terminal value is the negative terminal routing cost:

$$
V _ {T + 1} (s _ {T + 1}) = - C _ {T} (B _ {T}), \quad s _ {T+1}=B_T. \tag {3}
$$

At epoch $t$ , the state records the bookings already accepted before the current request:

$$
s _ {t} = B _ {t - 1}. \tag {4}
$$

If a passenger arrives, the system observes the request information $\xi _ t=(h_t,g_t)$ , where $h_t$ denotes the home-pickup location and $g_t$ denotes the passenger segment. Conditional on this realized request, the recommendation module generates the internal option set $O _ { t } = \{ h_t \} \cup M_t$ , and the operator chooses the price-adjustment vector $\pmb { a } _ { t } = ( a _ { t k } ) _ { k \in O _ { t } }$ from the feasible price set. The passenger then chooses among the internal options and the outside option according to the MNL model in Section 3.4.

The transition is option-specific. If the passenger chooses internal option $k \in O_t$ , the next state is

$$
s _ {t+1}^{(k)} = s_t \cup \{(t,k)\}.
$$

If the passenger chooses the outside option, or if no passenger arrives at epoch $t$ , no booking is added and the next state is $s _ {t+1}^{(0)}=s_t$ .

Because terminal value accounts for downstream routing costs, the immediate reward at epoch $t$ is the fare collected from the current passenger. Conditional on a realized arrival, the expected one-period reward is

$$
r _ {t} \left(\boldsymbol {a} _ {t}; \xi_t\right) = \sum_ {k \in O _ {t}} \left(f _ {e} + a _ {t k}\right) P _ {t k} \left(\boldsymbol {a} _ {t}\right). \tag {5}
$$

The outside option and the no-arrival event produce zero immediate reward. The Bellman optimality equation is therefore

$$
\begin{aligned}
V _ {t} (s _ {t}) ={}& (1-\lambda_t) V _ {t+1}(s_t) \\
&+ \lambda_t \mathbb{E}_{\xi_t} \left[ \max_{\boldsymbol {a}_t} \left\{
\sum_{k \in O_t} P_{tk}(\boldsymbol {a}_t) \left[ f_e + a_{tk} + V_{t+1}\left(s_{t+1}^{(k)}\right) \right]
+ P_{t0}(\boldsymbol {a}_t) V_{t+1}(s_t)
\right\} \right].
\end{aligned} \tag {6}
$$

Equation (6) separates the three possible transitions: acceptance through an internal option $k$ , outside-option choice, and no arrival. Solving it exactly is computationally demanding because the state space expands with the accepted booking set and because the value of an internal option depends on future demand and final route optimization. Consequently, we use approximate dynamic programming and represent the downstream value of each boarding option through an option-specific marginal cost. This approximation leads to the per-passenger expected-profit formulation introduced in Section 4 and derived in Appendix B.

# 3.4. Passenger choice model

Passengers' behavior is analyzed using a multinomial logit model (MNL). Once a passenger arrives at epoch $t$ , the operator presents an internal option set $O _ { t }$ together with the price-adjustment vector $\pmb { a } _ { t } = ( a _ { t k } ) _ { k \in O _ { t } }$ . The passenger then chooses one internal option or the outside option.

Passenger preferences are characterized by positive disutility weights for walking distance, price, and in-vehicle travel time, with corresponding parameters $\beta ^ { \mathrm { walk } }$, $\beta ^ { \mathrm { price } }$, and $\beta ^ { \mathrm { i v t } }$ . The price term enters utility as $- \beta ^ { \mathrm { p r i c e } } a _ { t k }$ with $\beta ^ { \mathrm { p r i c e } } > 0$ , so a positive price adjustment lowers utility and a negative adjustment (a discount) raises utility. The outside option is characterized by a common baseline utility constant $V _ { 0 }$ .

For a passenger, the utility of internal option $k \in O _ { t }$ is

$$
U _ {t k} = - \beta^ {\mathrm {i v t}} \tau_ {t k} - \beta^ {\mathrm {w a l k}} d _ {t k} - \beta^ {\mathrm {p r i c e}} a _ {t k} + \varepsilon_ {t k}, \tag {7}
$$

where $\tau _ { t k }$ is the displayed in-vehicle travel-time estimate, $d _ { t k }$ is the walking distance, and $a _ { t k }$ is the option-specific price adjustment. Because $\beta ^ { \mathrm { i v t } } > 0$ , $\beta ^ { \mathrm { w a l k } } > 0$ , and $\beta ^ { \mathrm { p r i c e } } > 0$ , shorter travel time, shorter walking distance, and a lower price adjustment all contribute to higher utility.

The disturbance terms $\varepsilon _ { t k }$ and $\varepsilon _ { t 0 }$ are posited to be independently and identically distributed following the Gumbel distribution, which gives rise to the conventional multinomial logit (MNL) framework. The outside option for a passenger is captured by

$$
U _ {t 0} = V _ {0} + \varepsilon_ {t 0}. \tag {8}
$$

The probability that a passenger chooses internal option $k$ is

$$
P _ {t k} \left(\boldsymbol {a} _ {t}\right) = \frac {\exp \left(- \beta^ {\mathrm {i v t}} \tau_ {t k} - \beta^ {\mathrm {w a l k}} d _ {t k} - \beta^ {\mathrm {p r i c e}} a _ {t k}\right)}{\exp \left(V _ {0}\right) + \sum_ {j \in O _ {t}} \exp \left(- \beta^ {\mathrm {i v t}} \tau_ {t j} - \beta^ {\mathrm {w a l k}} d _ {t j} - \beta^ {\mathrm {p r i c e}} a _ {t j}\right)}, \tag {9}
$$

and the probability of choosing the outside option is

$$
P _ {t 0} \left(\boldsymbol {a} _ {t}\right) = \frac {\exp \left(V _ {0}\right)}{\exp \left(V _ {0}\right) + \sum_ {j \in O _ {t}} \exp \left(- \beta^ {\mathrm {i v t}} \tau_ {t j} - \beta^ {\mathrm {w a l k}} d _ {t j} - \beta^ {\mathrm {p r i c e}} a _ {t j}\right)}. \tag {10}
$$

These probabilities satisfy

$$
P _ {t 0} (\boldsymbol {a} _ {t}) + \sum_ {k \in O _ {t}} P _ {t k} (\boldsymbol {a} _ {t}) = 1. \tag {11}
$$

The MNL model bridges the gap between pricing and routing outcomes. By offering larger discounts for meeting-point options and applying surcharges for home pickup, the operator can leverage prices to steer boarding decisions, thereby influencing the downstream routing cost.

# 4. Methodology

Building on the decision environment in Section 3, this section describes the tractable DRPO implementation. The framework combines meeting-point recommendation, option-level prediction of marginal insertion costs and service-time proxies, and a pricing module trained with a decision-focused objective. The goal is to approximate the downstream routing impact of current booking decisions while preserving an online `recommend -> predict -> price` workflow.

# 4.1. Framework overview

The main hurdle in dynamic routing pricing is that the downstream routing impact of assigning a passenger to a boarding location is not observable when prices must be quoted. DRPO therefore separates the offline label-construction task from the online prediction and pricing task. Online, DRPO follows a recommend -> predict -> price pipeline. First, the recommendation module constructs the feasible option set $O _ { t }$ by retrieving nearby meeting points for the current home location and adding the home-pickup option. Second, for each candidate option $k \in O _ { t }$ , the prediction module maps the state-option encoding $\phi ( s _ { t } , k )$ to two outputs: the predicted counterfactual marginal-cost proxy $\hat { c } _ { t k }$ and the service-position proxy $\widehat { q } _ { t k }$ used to compute the displayed in-vehicle travel time $\tau _ { t k }$ . Third, conditional on the offered set, the predicted costs, and the displayed service attributes, the pricing module determines the option-specific price vector $\mathbf { a } _ { t }$ to maximize expected profit. Offline, simulated booking episodes generate the counterfactual cost labels $c _ { t k }$ defined in Algorithm 1 and Appendix B, and these labels train the prediction model.

Fig. 4. Overview of the DRPO framework. The online layer recommends feasible boarding options, predicts option-level routing impact, and solves the pricing problem for the current passenger. The offline layer generates counterfactual routing labels and updates the predictive model with pointwise and decision-focused losses.

To obtain a tractable online decision problem, the dynamic program is reformulated as a per-passenger expected-profit problem in which the routing impact of each candidate option is represented by its marginal cost. Following Yang et al. (2016), the derivation from the Bellman equation is presented in Appendix B. Algorithm 1 summarizes the implementation workflow used in the experiments.

# Algorithm 1. Offline-online training and deployment procedure of DRPO

1. Precompute, for each home location, the adjacency set of the 10 nearest meeting points and initialize the choice-model and cost parameters.   
2. Simulate booking episodes under the calibrated demand model and, for each decision epoch $t$, enumerate the feasible boarding options $O _ { t } = \{ h _ { t } \} \cup M _ { t }$ .

3. For every candidate option $k \in O _ { t }$ , construct the option-level encoding $\phi ( s _ { t } , k )$ . To generate the counterfactual routing label, first form the baseline terminal booking set $B _ { T } ^ { - t }$ by removing any accepted booking associated with passenger $t$ , if present, and then form $B _ { T } ^ { t , k } = B _ { T } ^ { - t } \cup \{ ( t , k ) \}$ . The label is computed as $c _ { t k } = C _ { T } ( B _ { T } ^ { t , k } ) - C _ { T } ( B _ { T } ^ { - t } )$ using the same terminal routing solver, namely a greedy nearest-insertion heuristic with 2-opt post-processing. For each passenger, this requires one shared baseline routing evaluation and one counterfactual routing evaluation per candidate option, taking approximately 0.3,s per solve on the experimental hardware.   
4. Train the marginal-cost predictor first with the Huber loss for pointwise marginal-cost stability; after this warm-up stage, gradually increase the $\mathrm { S P O + }$ weight and optimize the blended objective in Eq. (32) to account for downstream pricing quality.   
5. At deployment time, retrieve the feasible meeting-point set for the current passenger, predict option-level marginal costs and travel-time proxies, solve the prediction-based pricing problem in Eq. (14), and display the resulting menu to the passenger.

# 4.2.1. State encoding and training data generation

The first step in the prediction pipeline is to convert the current DRT state and candidate option into a machine-learning input. We use the option-level encoding $\phi ( s _ { t } , k )$ , which combines the current accepted-booking state $s_t$ with the candidate boarding option $k$ . Spatially, the service area is partitioned into discrete cells; temporally, the booking horizon is divided into $T$ equal-length intervals. Passenger arrivals and accepted bookings are aggregated into this spatiotemporal representation so that the predictor can learn how current option choices relate to later routing burden. This encoding process is illustrated in Fig. 5, where boarding locations are represented as dots and the corresponding arrival-time interval for each passenger is indicated by the number within the dot.

While the precise daily passenger volume remains unknown, the spatiotemporal distribution of arrivals—as captured by the above encoding—can be predicted using CNN-based models. Both the spatial discretization granularity and the number of time intervals are tunable parameters in this framework.

![](images/2b96730de49874e0b7c800b18db3bd76be356e64ce4092430eda266e98b0ad62.jpg)

Fig. 5. State encoding.

To train the CNN, we construct option-level marginal-cost labels using the same counterfactual definition as Algorithm 1. For each simulated arrival $t$ and each candidate option $k \in O _ { t }$ , let $B _ { T } ^ { - t }$ denote the terminal booking set after removing any booking associated with passenger $t$ , and let $B _ { T } ^ { t , k } = B _ { T } ^ { - t } \cup \{ ( t , k ) \}$ denote the counterfactual terminal booking set in which passenger $t$ is assigned to option $k$ . We solve the terminal routing problem for both booking sets with the same routing solver and define $c _ { t k } = C _ { T } ( B _ { T } ^ { t , k } ) - C _ { T } ( B _ { T } ^ { - t } )$ . This label measures the incremental terminal routing cost of assigning passenger $t$ to option $k$ relative to not accepting passenger $t$ , so it is well defined for both the realized choice and unchosen candidate options. The training data therefore contain state-option pairs for all candidate options observed in simulation, which broadens coverage of the option-level cost differences encountered by the online pricing policy. The data generation procedure during simulation is depicted in Fig. 6.

![](images/b7d491e5dcc3c2f3da9a62656ba725a45c97e74001255021ea5d978450681174.jpg)  
(a) All passengers

![](images/7b89592b46a73e9cf4197c4fa83c0fa76a59e03faecccff70cf5ea36f5310872.jpg)  
(b) Excluding passenger 1

![](images/8673b4ab8f82b9d115a1affe8f6d3c322a40e6fa000b5c5a2a1bbc973bb014f8.jpg)  
(c) Excluding passenger 2

![](images/d4d40bf88d4e0153f33ff454e3d7b9a540501a9ac691ae63e4e63116595df92a.jpg)  
(d) Excluding passenger 3   
Fig. 6. Training data generation process.

# 4.2.2. Cost and in-vehicle travel time prediction module

When a new passenger arrives, the system evaluates each candidate option $k \in O_t$ using the same input representation $\phi(s_t,k)$ . The prediction module has two outputs. The first output, $\hat{c}_{tk}$ , is the online estimate of the counterfactual marginal-cost label $c_{tk}$ defined in Section 4.2.1. This predicted cost is the quantity passed to the pricing module in Section 4.2.3. The second output, $\widehat{q}_{tk}$ , is a service-position proxy used to estimate the displayed in-vehicle travel time $\tau_{tk}$ that enters the passenger choice model.

To accomplish this dual-output prediction task, we employ a CNN architecture consisting of two convolutional layers, one average pooling layer, and two fully connected layers. The model is trained on simulated state-option pairs. For each pair, the cost target is the offline counterfactual label $c_{tk}$, while the service-position target is derived from the simulated terminal route and used to construct the travel-time proxy shown to passengers.

The network processes the encoded input $\phi ( s _ { t } , k )$ through shared convolution and pooling layers and then uses two output branches: one branch predicts $\hat { c } _ { t k }$ and the other predicts $\widehat { q } _ { t k }$ . The core prediction mapping is

$$
\left(\hat {c} _ {t k}, \hat {q} _ {t k}\right) = \mathcal {N} _ {\theta} \left(\phi \left(s _ {t}, k\right)\right), \tag {12}
$$

where $\mathcal { N } _ { \theta }$ represents a network with trainable parameters $\theta$ . The first output $\hat { c } _ { t k }$ estimates the offline counterfactual cost label $c _ { t k }$ and is passed to the pricing optimization module. The second output $\hat { q } _ { t k }$ is not a cost; it is a service-position proxy used to construct $\tau _ { t k }$ for the MNL utility in Section 3.4.

The displayed in-vehicle travel time $\tau _ { t k }$ is a service attribute rather than a pricing cost. It is computed from the predicted service-position proxy $\widehat { q } _ { t k }$ together with preset speed and distance information. The system then presents each recommended boarding option with its price, walking distance, and predicted in-vehicle travel time, allowing the passenger to evaluate the menu before the final route is known.

# 4.2.3. Pricing optimization based on predicted costs

The predicted marginal insertion cost is then used to set option-specific price adjustments. The sign convention follows Section 3.4: $a _ { t k } > 0$ is a surcharge that increases the fare and lowers MNL utility, whereas $a _ { t k } < 0$ is a discount that decreases the fare and raises MNL utility. Thus the same price adjustment affects both the operator revenue term $f_e+a_{tk}$ and the passenger choice probability $P_{tk}(\pmb a_t)$ .

For a passenger arriving at epoch $t$ with internal option set $O_t$ , the ideal true-cost pricing benchmark assumes that the counterfactual marginal-cost labels $c_{tk}$ are known. The operator chooses the price-adjustment vector $\pmb { a } _ { t } = ( a _ { t k } ) _ { k \in O _ { t } }$ from the feasible price-adjustment set $\mathcal{A}_t$ . Under the MNL probabilities in Eq. (9), the expected-profit objective is
$$
\max  _ {\boldsymbol {a} _ {t}} \sum_ {k \in O _ {t}} \left(f _ {e} + a _ {t k} - c _ {t k}\right) P _ {t k} \left(\boldsymbol {a} _ {t}\right), \tag {13}
$$

where $P _ { t k } ( \pmb { a } _ { t } )$ is the MNL probability that passenger $t$ chooses option $k$ . The term $f _ { e } + a _ { t k } - c _ { t k }$ is the operator's net contribution if option $k$ is chosen: base fare plus price adjustment minus the option-level marginal-cost label.

In real-time operation, $c _ { t k }$ is unavailable because it depends on future requests and the final terminal route. DRPO therefore replaces $c _ { t k }$ with the prediction $\hat { c } _ { t k }$ from Section 4.2.2, yielding the deployed prediction-based pricing problem:

$$
\max  _ {\boldsymbol {a} _ {t}} \sum_ {k \in O _ {t}} \left(f _ {e} + a _ {t k} - \hat {c} _ {t k}\right) P _ {t k} \left(\boldsymbol {a} _ {t}\right). \tag {14}
$$

For compact notation, we suppress the passenger and time indices. Let $O$ denote the internal option set, $\pmb { a } = ( a _ { k } ) _ { k \in O }$ the price-adjustment vector, and $\pmb { c } = ( c _ { k } ) _ { k \in O }$ and $\hat { \pmb { c } } = ( \hat { c } _ { k } ) _ { k \in O }$ the true and predicted marginal-cost vectors. Let $P _ { k } ( { \pmb a } )$ be the MNL choice probability induced by the same utility specification as Eq. (9). The true-cost and prediction-based objectives can then be written as

$$
R (\boldsymbol {a}; \boldsymbol {c}) = \sum_ {k \in O} \left(f _ {e} + a _ {k} - c _ {k}\right) P _ {k} (\boldsymbol {a}), \tag {15}
$$

and

$$
R (\boldsymbol {a}; \hat {\boldsymbol {c}}) = \sum_ {k \in O} \left(f _ {e} + a _ {k} - \hat {c} _ {k}\right) P _ {k} (\boldsymbol {a}), \tag {16}
$$

The ideal benchmark and deployed prediction-based pricing problems are therefore

$$
\max  _ {\boldsymbol {a} \in \mathcal {A}} R (\boldsymbol {a}; \boldsymbol {c}), \quad \max  _ {\boldsymbol {a} \in \mathcal {A}} R (\boldsymbol {a}; \hat {\boldsymbol {c}}), \tag {17}
$$

where $\mathcal { A }$ denotes the feasible price-adjustment set, including operational lower and upper bounds on discounts and surcharges. In the experiments, this set imposes $-10 \leq a_{tk} \leq 10$ for each internal option. Appendix C derives the unconstrained structural characterization of the optimizer and shows how the optimal price adjustments depend on the cost vector. In the implementation, however, the online policy solves the bounded problem over $\mathcal { A }$ numerically, using the Appendix C characterization only as a structural guide. For any cost vector $\pmb { c }$ , let

$$
\boldsymbol {a} ^ {*} (\boldsymbol {c}) \in \underset {\boldsymbol {a} \in \mathcal {A}} {\operatorname{argmax}} R (\boldsymbol {a}; \boldsymbol {c}) \tag {18}
$$

denote the optimal pricing decision. Likewise, for a predicted cost vector $\widehat{\pmb{c}}$, let

$$
\boldsymbol {a} ^ {*} (\hat {\boldsymbol {c}}) \in \underset {\boldsymbol {a} \in \mathcal {A}} {\operatorname{argmax}} R (\boldsymbol {a}; \hat {\boldsymbol {c}}). \tag {19}
$$

The mapping $\pmb { c } \mapsto \pmb { a } ^ { * } ( \pmb { c } )$ serves as the downstream pricing oracle in DRPO: for any cost vector, it returns the feasible price-adjustment vector that optimizes the bounded MNL pricing problem in Eq. (18). In implementation, this oracle is evaluated over the feasible price-adjustment set $\mathcal { A }$ rather than by replacing the bounded problem with an unconstrained closed-form rule. The central learning question is therefore whether predicted costs induce high-quality pricing decisions, not only whether they minimize pointwise prediction error.

# 4.2.4. SPO decision-focused learning objective

The prediction module is not optimized as an isolated forecasting task. In DRPO, the predicted marginal-cost vector is consumed by the pricing problem in Section 4.2.3, and small prediction errors can change the selected price-adjustment vector when options have similar downstream value. We therefore use the Smart Predict-Then-Optimize (SPO) framework of Elmachtoub and Grigas (2022) to train the predictor with respect to the quality of the pricing decisions induced by its predictions.

To apply SPO/SPO+ to the bounded MNL pricing problem, we rewrite the objective in a linear canonical form by absorbing the nonlinear price-choice terms into a lifted decision vector ${ \pmb w } ( { \pmb a } )$ . For a fixed passenger and candidate option set $O$, let $\pmb { c } = ( c _ { k } ) _ { k \in O }$ and $\hat { \pmb { c } } = ( \hat { c } _ { k } ) _ { k \in O }$ denote the true and predicted option-level marginal-cost vectors. The true-cost downstream pricing benchmark is:

$$
\max  _ {\boldsymbol {a} \in \mathcal {A}} R (\boldsymbol {a}; \boldsymbol {c}). \tag {20}
$$

We define

$$
R _ {0} (\boldsymbol {a}) = \sum_ {k \in O} \left(f _ {e} + a _ {k}\right) P _ {k} (\boldsymbol {a}), \quad \boldsymbol {P} (\boldsymbol {a}) = \left[ P _ {k} (\boldsymbol {a}) \right] _ {k \in O}, \tag {21}
$$

so that

$$
R (\boldsymbol {a}; \boldsymbol {c}) = R _ {0} (\boldsymbol {a}) - \boldsymbol {c} ^ {\top} \boldsymbol {P} (\boldsymbol {a}). \tag {22}
$$

We then introduce the augmented cost vectors and lifted decision vector

$$
\boldsymbol {c} _ {a u g} = \left[ \begin{array}{l} \boldsymbol {c} \\ 1 \end{array} \right], \quad \hat {\boldsymbol {c}} _ {a u g} = \left[ \begin{array}{l} \hat {\boldsymbol {c}} \\ 1 \end{array} \right], \quad \boldsymbol {w} (\boldsymbol {a}) = \left[ \begin{array}{l} \boldsymbol {P} (\boldsymbol {a}) \\ - R _ {0} (\boldsymbol {a}) \end{array} \right]. \tag {23}
$$

For clarity, let

$$
\boldsymbol {y} = \boldsymbol {c} _ {a u g}, \quad \hat {\boldsymbol {y}} = \hat {\boldsymbol {c}} _ {a u g}. \tag {24}
$$

Then, for any feasible $\pmb { a } \in \mathcal { A }$

$$
\boldsymbol {y} ^ {\top} \boldsymbol {w} (\boldsymbol {a}) = \boldsymbol {c} ^ {\top} \boldsymbol {P} (\boldsymbol {a}) - R _ {0} (\boldsymbol {a}) = - R (\boldsymbol {a}; \boldsymbol {c}). \tag {25}
$$

Hence, maximizing downstream profit is equivalent to minimizing a linear objective:

$$
\max  _ {\boldsymbol {a} \in \mathcal {A}} R (\boldsymbol {a}; \boldsymbol {c}) \Leftrightarrow \min  _ {\boldsymbol {a} \in \mathcal {A}} \boldsymbol {y} ^ {\top} \boldsymbol {w} (\boldsymbol {a}). \tag {26}
$$

Let $\mathcal { W } = \operatorname{conv}\{ \pmb { w } ( \pmb { a } ) : \pmb { a } \in \mathcal { A } \}$ be the convex hull of lifted decisions generated by feasible price adjustments. Because the canonical objective is linear in $\pmb { w }$, optimizing over this convex hull preserves the same optimal value as optimizing over the original lifted feasible decisions. The set $\mathcal { W }$ is nonempty and bounded because $\pmb { P } ( \pmb { a } )$ lies in the probability simplex and $R _ { 0 } ( \pmb { a } )$ is bounded when $\pmb { a } \in \mathcal { A }$. This gives the bounded pricing oracle a standard SPO-compatible minimization form:

$$
z ^ {*} (\boldsymbol {y}) = \underset {\boldsymbol {w} \in \mathcal {W}} {\operatorname{min}} \boldsymbol {y} ^ {\top} \boldsymbol {w}, \quad \boldsymbol {w} ^ {*} (\boldsymbol {y}) \in \operatorname{argmin} _ {\boldsymbol {w} \in \mathcal {W}} \boldsymbol {y} ^ {\top} \boldsymbol {w}. \tag {27}
$$

Here, $z ^ { \ast } ( \pmb { y } )$ denotes the optimal objective value under augmented cost vector $\pmb { y }$, while $\pmb { w } ^ { * } ( \pmb { y } )$ denotes an optimal lifted decision achieving that value. Operationally, the oracle solves the original bounded pricing problem over $\mathcal { A }$ from Section 4.2.3 and then maps the resulting price-adjustment vector into its lifted representation. Thus the lifted formulation is a canonical SPO representation of the same downstream pricing decision, not a separate pricing model.

The SPO loss quantifies the downstream regret associated with pricing strategies that rely on predicted rather than actual cost vectors.

$$
\ell_ {S P O} (\hat {\boldsymbol {y}}, \boldsymbol {y}) = \boldsymbol {y} ^ {\top} \boldsymbol {w} ^ {*} (\hat {\boldsymbol {y}}) - z ^ {*} (\boldsymbol {y}). \tag {28}
$$

Using Eq. (26), this can be written directly as the profit loss induced by prediction error:

$$
\ell_ {S P O} (\hat {\boldsymbol {y}}, \boldsymbol {y}) = R \left(\boldsymbol {a} ^ {*} (\boldsymbol {c}); \boldsymbol {c}\right) - R \left(\boldsymbol {a} ^ {*} (\hat {\boldsymbol {c}}); \boldsymbol {c}\right). \tag {29}
$$

Because $\ell _ { S P O }$ is generally nonconvex and nondifferentiable with respect to the predictions, we use the convex surrogate loss $\ell _ { S P O + }$ introduced by Elmachtoub and Grigas (2022). In the lifted pricing representation, the surrogate is:

$$
\ell_ {S P O +} (\hat {\boldsymbol {y}}, \boldsymbol {y}) = \max  _ {\boldsymbol {w} \in \mathcal {W}} (\boldsymbol {y} - 2 \hat {\boldsymbol {y}}) ^ {\top} \boldsymbol {w} + 2 \hat {\boldsymbol {y}} ^ {\top} \boldsymbol {w} ^ {*} (\boldsymbol {y}) - z ^ {*} (\boldsymbol {y}). \tag {30}
$$

A valid subgradient with respect to the augmented prediction is:

$$
2 \left(\boldsymbol {w} ^ {*} (\boldsymbol {y}) - \boldsymbol {w} ^ {*} (2 \hat {\boldsymbol {y}} - \boldsymbol {y})\right) \in \partial_ {\hat {\boldsymbol {y}}} \ell_ {S P O +} (\hat {\boldsymbol {y}}, \boldsymbol {y}). \tag {31}
$$

Because the final element of $\widehat { \pmb { y } }$ is fixed at 1, backpropagation is applied only to the first $| O |$ components associated with $\widehat{\pmb{c}}$. Although DRPO makes pricing decisions sequentially online, predictor training is performed offline from recorded decision epochs. At each epoch $t$, the neural network maps the encoded state-option features $\phi ( s _ { t } , k )$ to the option-specific predicted marginal-cost vector $\hat { \pmb { c } }$, which is passed to the bounded pricing oracle to obtain ${ \pmb a } ^ { * } ( \hat { \pmb c } )$. After the episode, the realized counterfactual cost vector $\pmb{c}$ is reconstructed from final-route data, paired with $( \widehat { \pmb { y } } , \pmb { y } )$, and used to update the prediction network.

The training objective combines a pointwise prediction term and a decision-focused term. The Huber term fits the reconstructed option-level marginal-cost labels and stabilizes the state-option representation. The $\mathrm { S P O + }$ term evaluates predictions through the downstream pricing oracle, so prediction errors are penalized according to their effect on the selected price-adjustment vector and the associated pricing objective. The blended objective is

$$
\mathcal {L} (\theta) = \alpha_ {S P O +} \ell_ {S P O +} (\widehat {\boldsymbol {y}}, \boldsymbol {y}) + (1 - \alpha_ {S P O +}) \mathcal {L} _ {H u b e r} (\widehat {\boldsymbol {c}}, \boldsymbol {c}). \tag {32}
$$

The weight $\alpha _ { S P O + } \in [ 0 , 1 ]$ controls the transition from pointwise prediction training to decision-focused training. During the warm-up stage, $\alpha _ { S P O + } = 0$ and the model is trained only with the Huber loss. After the cost-prediction representation has stabilized, $\alpha _ { S P O + }$ is increased gradually so that the SPO+ term contributes to updates while the Huber term continues to regularize pointwise accuracy. This schedule is intended to balance stable learning with downstream pricing quality. The resulting performance should be interpreted as the outcome of the full DRPO pipeline--recommendation, prediction, and pricing--with SPO+ used to align the prediction module with the pricing decision.

# 5. Results

This section evaluates whether the DRPO framework improves many-to-one DRT operations through integrated recommendation, prediction, and pricing. The synthetic benchmark provides controlled evidence across baseline strategies, the Beijing commuter-corridor case provides semi-realistic evidence, and the sensitivity analysis examines how behavioral assumptions affect operator and passenger outcomes.

# 5.1. Experimental design

We assess DRPO in a many-to-one DRT framework where passengers start from various locations and head to a single destination. Each strategy is evaluated under identical default conditions and passenger decision-making parameters, including the option to opt out, so passenger abandonment remains endogenous to the offered menu and prices. Sections 5.2 and 5.3 report provider-side metrics, including total cost and net profit, and passenger-side metrics, including home-pickup share and quit rate. Section 5.4 examines how these outcomes change under different behavioral assumptions.

The benchmark set is designed to identify the role of recommendation and pricing at the policy level, rather than to isolate every internal DRPO component. Only-home assigns all accepted passengers to home pickup and provides a convenience-focused operating reference. Only-meeting-points assigns passengers to recommended meeting points and provides an efficiency-focused reference that may be unattractive because of high passenger abandonment. No-pricing allows passengers to choose between home pickup and recommended meeting points without price differentiation, revealing the boarding-location choices induced by service attributes alone. Static-pricing adds fixed discounts for meeting points and fixed surcharges for home pickup, measuring the effect of a simple non-adaptive incentive rule. DSPO, based on Akkerman et al. (2024), provides the closest learning-based pricing benchmark. DRPO is the proposed integrated `recommend -> predict -> price` framework with decision-focused prediction training.

These comparisons support different levels of evidence. The simple baselines show the operational tradeoff between home pickup, meeting-point steering, and passenger retention. Static-pricing shows whether a non-adaptive incentive rule is sufficient. The DSPO comparison evaluates the implemented DRPO framework against a learning-based pricing benchmark. Because the experiments do not include a complete factorial ablation of recommendation, prediction, pricing, and SPO+ training, DRPO-vs-DSPO differences should be interpreted as evidence for the integrated framework rather than as the isolated effect of any single component. Section 5.2 studies these comparisons in the controlled synthetic benchmark, and Section 5.3 reports the semi-realistic Beijing simulation.

# 5.2. Synthetic case

We use a synthetic case study to examine the basic properties of the DRPO framework.

# 5.2.1. Problem setting

Using the Gehring and Homberger (2002) benchmark, we generated synthetic demand data comprising 200 geographic locations. These locations were randomly partitioned into mutually exclusive training and test sets, each containing 100 locations. From each set, we randomly sampled 10 locations as candidate recommended meeting points. The resulting spatial distribution is shown in Fig. 7.

Driving times are calculated from Euclidean distances. Vehicle speed is set to 30 km/h, driver wage to $\yen 30$ per hour, and fuel cost to $\yen 0.6$ per distance unit.

![](images/107b07f7a4ca6a4d5356905ff2dbd2fa9cc101f011ea568abc7b440b18ade50b.jpg)  
Fig. 7. Spatial distribution of home locations and candidate meeting points in the synthetic case.

In the experimental setup, dwell time is modeled as a spatially heterogeneous variable, with distribution parameters derived from the six-hump camel-back function, a standard multimodal test function in the optimization literature.

This construction reflects the intuition that dwell times may be higher in high-density residential areas than in rural areas or suburban stops. Dwell time at location $i$ is obtained by projecting the service area onto the domain $( x \in [ - 3 , 3 ] , y \in [ - 2 , 2 ] )$ of

$$
l _ {i} (x, y) = \left(4 - 2. 1 x ^ {2} + \frac {x ^ {4}}{3}\right) x ^ {2} + x y + (- 4 + 4 y ^ {2}) y ^ {2}, \tag {33}
$$

which is an optimization function proposed in Molga and Smutnicki (2005). We bound the dwell time on $l _ { i } \in [ 1 , 1 0 ]$ minutes.

The daily number of arriving passengers follows a negative binomial distribution $D$ , where the probability mass function is given by

$$
P r (D = d) = (\frac {d + r - 1}{d}) (1 - p) ^ {d} p ^ {r}, \quad d = 0, 1, 2, \dots , \tag {34}
$$

where $r = 9 0$ , $p = 0 . 5$ , and $\begin{array} { r } { \mathbb { E } [ D ] = \frac { r ( 1 - p ) } { p } = 9 0 ~ } \end{array}$ . Passenger arrival times are uniformly distributed over the service period. Daily passenger capacity is constrained by a fleet of 12 buses, each with 12 seats. Because actual passenger choice data are not available, passenger decisions are modeled using the MNL model in Section 3.4.

# 5.2.2. Analysis of results

We set the maximum number of episodes to 200 for all strategies. Table 1 reports the corresponding averages over 30 random seeds.

Table 1. Results for the synthetic case.

| Strategy | Home-pickup share | Quit rate | Total cost | Cost saving vs. Only-home | Net profit |
|---|---:|---:|---:|---:|---:|
| Only-home | 99.00% | 1.01% | 2890.75 | 0.00% | 1590.92 |
| Only-meeting-points | 0.00% | 45.23% | 702.95 | 75.68% | 1780.39 |
| No-pricing | 98.00% | 1.01% | 2885.99 | 0.16% | 1595.67 |
| Static-pricing | 22.00% | 1.20% | 2382.40 | 17.59% | 2090.93 |
| DSPO | 53.66% | 3.69% | 2218.46 | 23.26% | 2193.79 |
| DRPO | 45.32% | 3.11% | 2093.06 | 27.59% | 2346.10 |

Table 1 compares the benchmark policies in the synthetic case. Only-home and No-pricing yield nearly identical outcomes, suggesting that, in the absence of clear incentives, most travelers still choose home pickup. At the other extreme, Only-meeting-points has the lowest overall cost, but it also produces a $4 5 . 2 3 \%$ dropout rate, making it less attractive from an operational perspective despite its routing efficiency. Static-pricing improves on these simple references, reducing total cost by $1 7 . 5 9 \%$ relative to Only-home and increasing net profit to 2090.93.

Among the practical pricing strategies in Table 1, DRPO reports the lowest average total cost and the highest average net profit. Its average total cost is 2093.06, corresponding to a $2 7 . 5 9 \%$ reduction relative to Only-home and a $1 2 . 1 4 \%$ reduction relative to Static-pricing. Its mean net profit is 2346.10. Compared with DSPO, DRPO increases net profit by $6 . 9 4 \%$ and reduces total cost by $5 . 6 5 \%$ , while reducing the home-pickup share from $5 3 . 6 6 \%$ to $4 5 . 3 2 \%$ . In the paired 30-seed comparison, DRPO achieves a higher net profit than DSPO in 29 out of 30 cases. These findings provide controlled synthetic evidence that the integrated DRPO policy can shift more passengers toward meeting points while avoiding the high-quit outcome of the pure meeting-point policy. They should not be read as a complete decomposition of the contribution of recommendation, cost prediction, pricing, and SPO+ training.

# 5.3. Semi-realistic Yanjiao-Guomao case in Beijing

We conduct a semi-realistic simulation on the Yanjiao-Guomao commuter corridor in Beijing to examine DRPO in a spatial setting based on an actual commuter corridor and existing bus-stop locations.

# 5.3.1. Problem setting

The case uses real spatial context but modeled demand and passenger choice. Specifically, it focuses on the commuter route from the suburbs outside the Fifth Ring Road in Beijing to the Central business district (CBD). We use bus-stop data published by the Beijing Municipal Transportation Commission and select 170 existing bus stops as recommended meeting points.

Passenger demand is generated with parameters $r = 2 4 0$ and $p = 0 . 5$ , so that $\begin{array} { r } { \mathbb { E } [ D ] = { \frac { r ( 1 - p ) } { p } } = 2 4 0 } \end{array}$ . Then, we randomly generate 300 home locations, sampling them according to population density derived from official statistics for Beijing.

The system includes 25 buses, each with 12 seats. The fare structure includes a base fare $f _ { e }$ of 30 units, with dynamic price adjustments ranging from [−10, 10] units. As in the synthetic case, we set vehicle speed to $3 0 \ \mathrm { k m / h }$ , driver wage to $\yen 30/ h$ , and fuel cost to $\yen 0.6/\mathrm { k m }$ . Fig. 8 shows the spatial distribution.

![](images/0d37432644f208c903822dc13d14df529487c4b826e0bc6d571cffd45cf47234.jpg)  
Fig. 8. Spatial distribution of home locations and candidate meeting points in the semi-realistic Beijing case.

# 5.3.2. Analysis of results

Table 2 reports the performance of the DRPO framework relative to the benchmark strategies in the semi-realistic Beijing simulation.

Table 2. Results for the semi-realistic Beijing case.

| Strategy | Home-pickup share | Quit rate | Total cost | Cost saving vs. Only-home | Net profit |
|---|---:|---:|---:|---:|---:|
| Only-home | 99.26% | 0.74% | 5115.22 | 0.00% | 6644.78 |
| Only-meeting-points | 0.00% | 52.63% | 337.30 | 93.41% | 5272.70 |
| No-pricing | 98.18% | 0.74% | 5093.67 | 0.42% | 6666.33 |
| Static-pricing | 72.94% | 2.03% | 4298.90 | 15.96% | 7311.10 |
| DSPO | 45.35% | 4.04% | 3717.28 | 27.33% | 7656.22 |
| DRPO | 37.18% | 3.24% | 3667.18 | 28.31% | 7798.82 |

Table 2 compares the benchmark policies in the semi-realistic Beijing simulation. Only-home and No-pricing yield nearly identical outcomes, suggesting that, in the absence of clear incentives, most passengers still choose home pickup. Only-meeting-points produces the lowest total cost, but it also has a $5 2 . 6 3 \%$ dropout rate and the smallest net profit among the feasible pricing strategies, making it unattractive from an operational perspective despite its routing efficiency. Static-pricing improves on the non-adaptive references, but the learning-based strategies report lower total cost and higher net profit.

In the semi-realistic Beijing simulation, Table 2 reports that DRPO has an average total cost of 3667.18, corresponding to a $2 8 . 3 1 \%$ reduction relative to Only-home and a $1 . 3 5 \%$ reduction relative to DSPO. Its average net profit is 7798.82, which is $1 . 8 6 \%$ higher than DSPO. The home-pickup share decreases from $4 5 . 3 5 \%$ under DSPO to $3 7 . 1 8 \%$ under DRPO, and the quit rate decreases from $4 . 0 4 \%$ to $3 . 2 4 \%$ . These reported averages from the available matched simulation runs provide directional evidence that the integrated recommendation, prediction, and pricing policy can shift more passengers to designated meeting points without increasing abandonment in this case. As in the synthetic benchmark, this comparison evaluates the implemented framework as a whole.

Overall, the Beijing case provides semi-realistic directional evidence for applying DRPO in a commuter-corridor setting with real stop locations and simulated demand. It should be interpreted as simulation-based evidence rather than field validation or deployment evidence.

# 5.4. Sensitivity analysis

We use the synthetic benchmark to conduct behavioral comparative statics for the passenger choice model. The purpose is not to claim broad robustness across all operating environments, but to examine how the value of dynamic incentives changes when passenger responses to price, outside alternatives, and home pickup vary. Specifically, we vary the price-sensitivity parameter $\beta ^ { p r i c e }$ , the outside-option utility $U _ { 0 }$ , and the home-pickup alternative-specific constant $A S C _ { h o m e }$ .

# 5.4.1. Experimental settings

The sensitivity analysis follows a one-factor-at-a-time design: one behavioral parameter is varied while the other two remain at their default values. Key performance indicators include net profit, total operating cost, and

quit rate. These metrics are interpreted jointly, because a lower operating cost can reflect either more efficient routing or a loss of retained demand.

To capture passengers' intrinsic preference for home pickup, we incorporate $A S C _ { h o m e }$ as an alternative-specific constant for this option. A higher $A S C _ { h o m e }$ value represents a stronger preference for door-to-door service, while a lower value indicates greater willingness to walk to a recommended meeting point. The utility function for the home-pickup option is therefore modified as follows:

$$
U _ {t h _ {t} g} = A S C _ {h o m e} - \beta_ {g} ^ {i v t} \tau_ {t h _ {t}} - \beta_ {g} ^ {p r i c e} a _ {t h _ {t}} + \varepsilon_ {t h _ {t}}, \tag {35}
$$

while the utility of meeting-point options remains unchanged.

For each variable, we evaluate five levels while holding the other behavioral variables at their default settings $( \beta ^ { p r i c e } = 0 . 2 5$ , $U _ { 0 } = - 1 . 0$ , and $A S C _ { h o m e } = 1 . 4 )$ ). The price coefficient in the utility function is $- \beta ^ { p r i c e } = - 0 . 2 5$ . Each test uses DRPO, with three repetitions for each level.

# 5.4.2. Effect of price sensitivity

We first examine the effect of the price-sensitivity parameter $\beta ^ { p r i c e }$ . This parameter determines how strongly passengers respond to discounts on meeting points and surcharges for home pickup. Since the utility coefficient for price is $- \beta ^ { p r i c e }$ , a higher value of $\beta ^ { p r i c e }$ implies stronger price sensitivity and gives the operator more leverage to steer boarding choices through price adjustments.

Fig. 9 shows that increasing $\beta ^ { p r i c e }$ from 0.15 to 0.35 is associated with higher net profit and lower total operating cost. Specifically, net profit increases by $6 4 . 7 \%$ , while total operating cost decreases by $4 3 . 6 \%$ . This pattern indicates that dynamic incentives are more valuable when passengers are sufficiently price-sensitive to respond to meeting-point discounts and home-pickup surcharges.

![](images/64c8b089843070865ea7549999b235c2079d100480c690522ca8c87ed2b89942.jpg)  
Fig. 9. Sensitivity to the price-sensitivity parameter $\beta ^ { p r i c e }$ . The top panel reports relative changes in mean net profit and mean total cost from the default setting $\beta ^ { p r i c e } = 0 . 2 5$ ; the bottom panel reports the mean quit rate.

When passengers are less price-sensitive, the same pricing policy has less influence on boarding-location choices, and the operational value of dynamic pricing is correspondingly smaller. Price sensitivity is therefore a key behavioral condition for effective demand steering.

# 5.4.3. Effect of outside option utility

We next consider the outside-option utility $V _ { 0 }$ , which represents the attractiveness of not using the DRT service. A higher value of $V _ { 0 }$ means that alternative modes, such as private vehicles or conventional public transport, become more attractive relative to the offered DRT menu.

Fig. 10 shows that the outside-option utility primarily affects passenger retention. When $V _ { 0 } \leq - 1 . 0$ , the DRT service remains attractive and quit rates are low. As $V _ { 0 }$ increases to 0 or 1.0, the quit rate rises sharply: from $4 . 0 \%$ at $V _ { 0 } = - 1 . 0$ to $6 0 . 5 \%$ at $V _ { 0 } = 0 . 0$ and $9 5 . 2 \%$ at $V _ { 0 } = 1 . 0$ . Total operating cost also declines, reaching 286.3 at $V _ { 0 } = 1 . 0$ .

![](images/34016b754808792369b3af160701f8f52642c175d2e8d77f8038d4b90589597e.jpg)  
Fig. 10. Sensitivity to outside-option utility $V _ { 0 }$ . The top panel reports relative changes in mean net profit and mean total cost from the default setting $V _ { 0 } = - 1 . 0$ ; the bottom panel reports the mean quit rate.

These apparent cost savings require careful interpretation. When outside options are attractive, the decline in cost is driven largely by lost demand rather than improved routing efficiency. Therefore, low operating costs

should not be considered in isolation. They should be assessed together with demand-side metrics such as quit rate and passenger retention.

# 5.4.4. Effect of home-pickup preference

Finally, we examine the role of the home-pickup alternative-specific constant $A S C _ { h o m e }$ , which captures intrinsic preference for home pickup. A lower $A S C _ { h o m e }$ means that passengers place less additional utility on door-to-door pickup and are more willing to consider meeting-point options.

Fig. 11 shows that lower values of $A S C _ { h o m e }$ improve operator-side outcomes in this synthetic setting. As $A S C _ { h o m e }$ decreases from 1.8 to 1.0, net profit increases by $3 7 . 2 \%$ while total operating cost declines by $4 1 . 3 \%$ . This pattern suggests that meeting-point steering becomes easier when passengers have a weaker intrinsic preference for home pickup.

![](images/a869e96d4bb21251467168c2f03fb704d93e5f4e4350beaf74683b647ffa5115.jpg)

Fig. 11. Sensitivity to the home-pickup preference parameter $A S C _ { h o m e }$ . The top panel reports relative changes in mean net profit and mean total cost from the default setting $A S C _ { h o m e } = 1 . 4$ ; the bottom panel reports the mean quit rate.

At the same time, the result reveals a passenger-retention tradeoff. As $A S C _ { h o m e }$ decreases, the quit rate increases from $1 . 7 \%$ when $A S C _ { h o m e } = 1 . 8$ to $8 . 1 \%$ at $A S C _ { h o m e } = 1 . 0$ . Thus, weaker home-pickup preference creates more room for routing-efficiency gains, but it may also increase abandonment for some passengers.

# 5.4.5. Summary of sensitivity analysis

The sensitivity analysis complements the main experiments by clarifying the behavioral conditions under which dynamic incentives are most useful. Higher price sensitivity strengthens the operator's ability to steer boarding choices, a more attractive outside option can make lower operating costs reflect lost demand, and weaker home-pickup preference makes meeting-point steering easier but can increase quit rates. These results should therefore be interpreted as behavioral comparative statics, with operator-side outcomes evaluated jointly with passenger retention.

# 6. Conclusion

This study examines a dynamic demand-management problem that integrates meeting-point recommendation and pricing in many-to-one DRT systems. The central challenge is to offer passengers a menu of home-pickup and meeting-point options under sequential arrivals, while accounting for both passenger acceptance and downstream routing performance.

We address this challenge by formulating the setting as a Markov decision process and developing DRPO, a predict-then-optimize framework for Dynamic Meeting-Point Recommendation and Pricing Optimization. DRPO follows a `recommend -> predict -> price` sequence that combines feasible option generation, option-level marginal insertion cost prediction, and price optimization under passenger choice.

The results from the synthetic benchmark and the semi-realistic Beijing simulation provide evidence for the proposed approach under the tested simulation settings. In the synthetic scenario, DRPO achieves the highest mean net profit among the practical pricing strategies and reduces total cost by $2 7 . 6 \%$ compared to Only-home. In the Beijing case, DRPO reports the highest mean net profit among the strategies tested. It also reduces costs by $2 8 . 3 1 \%$ relative to Only-home and increases net profit by $1 . 8 6 \%$ over DSPO. The sensitivity analysis indicates that the operational value of dynamic incentives depends on passenger price sensitivity, and it underscores the importance of interpreting cost reductions jointly with quit behavior when outside options become more appealing.

The results reveal three key insights. First, dynamic incentives can guide passengers from home pickup to designated meeting points without pushing the system toward the high-exit outcome observed under a pure meeting-point policy. Second, explicit behavioral modeling is important for assessing both operational efficiency and passenger retention. Third, aligning marginal-cost prediction with the downstream pricing decision is useful within the full DRPO pipeline, although the reported gains should be interpreted as the performance of the integrated recommendation, prediction, and pricing framework rather than as an isolated effect of any single component.

The study provides a scalable, data-driven framework for integrating meeting-point recommendation and dynamic pricing in DRT systems. It offers a method for balancing efficiency, profitability, and passenger acceptance under uncertain demand and operational constraints. Future work can extend the framework by jointly designing meeting-point locations, considering many-to-many service settings, and incorporating additional constraints such as time windows or feeder-service network structures.

# Appendix A. Notation

| Symbol | Description |
|---|---|
| $\mathcal{G}$ | Set of passenger segments. |
| $g$ | Passenger-segment index. |
| $t$ | Decision epoch or passenger request index. |
| $T$ | Final booking epoch. |
| $h_t$ | Home-pickup option for passenger $t$. |
| $M_t$ | Recommended meeting-point set for passenger $t$. |
| $O_t$ | Internal option set offered to passenger $t$, $O_t=\{h_t\}\cup M_t$. |
| $0$ | Outside option. |
| $k$ | Internal boarding option index in $O_t$. |
| $B_t$ | Accepted booking set through epoch $t$. |
| $B_T$ | Final accepted booking set after the booking horizon. |
| $B_T^{-t}$ | Counterfactual terminal booking set with passenger $t$ removed. |
| $B_T^{t,k}$ | Counterfactual terminal booking set with passenger $t$ assigned to option $k$. |
| $R^T$ | Final route plan after the booking horizon. |
| $\mathcal{E}(R^T)$ | Arc set of the final route plan. |
| $\mathcal{N}(R^T)$ | Node set of the final route plan. |
| $c_\omega$ | Unit time-related operating cost. |
| $c_f$ | Unit distance-related operating cost. |
| $\omega_{ij}$ | Travel time on arc $(i,j)$. |
| $d_{ij}$ | Travel distance on arc $(i,j)$. |
| $l_i$ | Dwell or service time at node $i$. |
| $C_T(B_T)$ | Terminal routing cost associated with final bookings. |
| $\Pi(B_T)$ | Net profit associated with final bookings. |
| $f_e$ | Base fare. |
| $a_{tk}$ | Price adjustment for passenger $t$ and internal option $k$; positive values denote surcharges and negative values denote discounts. |
| $\boldsymbol{a}_t$ | Vector of option-specific price adjustments offered at epoch $t$. |
| $s_t$ | System state at epoch $t$. |
| $\xi_t=(h_t,g_t)$ | Observed request information when a passenger arrives at epoch $t$. |
| $s_{t+1}^{(k)}$ | Next state if passenger $t$ chooses internal option $k$. |
| $s_{t+1}^{(0)}$ | Next state if passenger $t$ chooses the outside option or no passenger arrives. |
| $V_t(s_t)$ | Value function at epoch $t$ under state $s_t$. |
| $\lambda_t$ | Arrival probability at epoch $t$. |
| $\mu_g$ | Probability that an arriving passenger belongs to segment $g$. |
| $r_t(\boldsymbol{a}_t)$ | Expected one-period reward induced by price decision $\boldsymbol{a}_t$. |
| $U_{tk}$ | Utility of internal option $k$ for passenger $t$. |
| $U_{t0}$ | Utility of the outside option for passenger $t$. |
| $P_{tk}(\boldsymbol{a}_t)$ | Probability that passenger $t$ chooses internal option $k$. |
| $P_{t0}(\boldsymbol{a}_t)$ | Probability that passenger $t$ chooses the outside option. |
| $d_{tk}$ | Walking distance from passenger $t$ to internal option $k$. |
| $\tau_{tk}$ | Displayed in-vehicle travel-time estimate for passenger $t$ under option $k$. |
| $\varepsilon_{tk}$, $\varepsilon_{t0}$ | Random utility terms for internal and outside options. |
| $\beta^{\mathrm{walk}}$ | Walking-distance utility coefficient. |
| $\beta^{\mathrm{price}}$ | Price utility coefficient. |
| $\beta^{\mathrm{ivt}}$ | In-vehicle travel-time utility coefficient. |
| $V_0$ | Baseline utility of the outside option. |
| $\phi(s_t,k)$ | State-option encoding used by the prediction module. |
| $\widehat{q}_{tk}$ | Predicted service-position proxy used to compute $\tau_{tk}$. |
| $c_{tk}$ | Counterfactual marginal insertion-cost label for passenger $t$ and option $k$, defined as $C_T(B_T^{t,k})-C_T(B_T^{-t})$. |
| $\widehat{c}_{tk}$ | Online prediction of the counterfactual marginal-cost label $c_{tk}$. |
| $\pmb{c}$ | Vector of true option-level cost proxies. |
| $\widehat{\pmb{c}}$ | Vector of predicted option-level cost proxies. |
| $\mathcal{A}$ | Feasible price-adjustment set. |
| $\boldsymbol{P}(\boldsymbol{a})$ | Choice-probability vector induced by price decision $\boldsymbol{a}$. |
| $R(\boldsymbol{a};\boldsymbol{c})$ | Downstream pricing objective evaluated under cost vector $\boldsymbol{c}$. |
| $R_0(\boldsymbol{a})$ | Cost-independent revenue component in the lifted SPO notation. |
| $\boldsymbol{c}_{\mathrm{aug}}$ | Augmented true cost vector used in the SPO formulation. |
| $\widehat{\boldsymbol{c}}_{\mathrm{aug}}$ | Augmented predicted cost vector used in the SPO formulation. |
| $\boldsymbol{y}$ | Compact augmented true-cost vector in the lifted SPO derivation. |
| $\widehat{\boldsymbol{y}}$ | Compact augmented predicted-cost vector in the lifted SPO derivation. |
| $\boldsymbol{w}(\boldsymbol{a})$ | Lifted decision vector induced by price decision $\boldsymbol{a}$. |
| $\mathcal{W}$ | Convex hull of feasible lifted decision vectors. |
| $z^*(\boldsymbol{y})$ | Optimal lifted objective value under augmented vector $\boldsymbol{y}$. |
| $\boldsymbol{w}^*(\boldsymbol{y})$ | Optimal lifted decision under augmented vector $\boldsymbol{y}$. |
| $\ell_{\mathrm{SPO}}$ | SPO regret loss. |
| $\ell_{\mathrm{SPO+}}$ | SPO+ convex surrogate loss. |
| $\mathcal{L}_{\mathrm{Huber}}$ | Huber prediction loss. |
| $\alpha_{\mathrm{SPO+}}$ | Weight assigned to the SPO+ term in the blended training loss. |

# Appendix B.

Starting from Eq. (6), condition on a realized passenger request $\xi_t=(h_t,g_t)$ at epoch $t$ . The internal option set is $O_t$ and the price vector is $\pmb { a } _ { t } = ( a _ { t k } ) _ { k \in O _ { t } }$ . The no-arrival term and the outside-option continuation term do not depend on which internal option is evaluated, so the action-dependent part of the Bellman equation can be written using option-specific next states.

If the passenger chooses internal option $k \in O _ { t }$ , the immediate reward is $f _ { e } + a _ { t k }$ and the next state is $s _ { t+1 }^{(k)} = s_t \cup \{(t,k)\}$ . Therefore, the value associated with internal option $k$ is

$$
\left(f _ {e} + a _ {t k}\right) + V _ {t + 1} \left(s _ {t+1}^{(k)}\right). \tag {B.1}
$$

If the passenger chooses the outside option, the state remains $s_t$ and the continuation value is $V _ { t + 1 } ( s _ { t } )$ . Relative to this unchanged-state continuation value, the net gain from internal option $k$ is

$$
[ f _ {e} + a _ {t k} + V _ {t + 1} (s _ {t+1}^{(k)}) ] - V _ {t + 1} (s _ {t}) = (f _ {e} + a _ {t k}) - [ V _ {t + 1} (s _ {t}) - V _ {t + 1} (s _ {t+1}^{(k)}) ]. \tag {B.2}
$$

We define the option-specific opportunity cost as

$$
\Delta V _ {k} \left(s _ {t}\right) = V _ {t + 1} \left(s _ {t}\right) - V _ {t + 1} \left(s _ {t+1}^{(k)}\right). \tag {B.3}
$$

This quantity measures the expected future-profit loss caused by accepting the current passenger through option $k$ instead of leaving the accepted booking set unchanged. Under the MNL model, the probability that the passenger chooses option $k$ under price vector $\pmb { a } _ { t }$ is $P _ { t k } ( \pmb { a } _ { t } )$ . The action-dependent expected net profit at epoch $t$ is therefore

$$
\sum_ {k \in O _ {t}} P _ {t k} \left(\boldsymbol {a} _ {t}\right) \left(f _ {e} + a _ {t k} - \Delta V _ {k} \left(s _ {t}\right)\right). \tag {B.4}
$$

Given $\Delta V _ { k } ( s _ { t } )$ , maximizing the conditional action-dependent part of the Bellman equation is equivalent to maximizing the following per-passenger expected net profit:

$$
\max  _ {\boldsymbol {a} _ {t}} \sum_ {k \in O _ {t}} P _ {t k} \left(\boldsymbol {a} _ {t}\right) \left(f _ {e} + a _ {t k} - \Delta V _ {k} \left(s _ {t}\right)\right). \tag {B.5}
$$

The true opportunity cost $\Delta V _ { k } ( s _ { t } )$ is not observable in real time, because it depends on all future arrivals and on the final routing plan. We approximate it by an option-level counterfactual marginal-cost label. For the observed request, let $B _ { T } ^ { - t }$ denote the terminal booking set without passenger $t$ and let $B _ { T } ^ { t , k } = B _ { T } ^ { - t } \cup \{ ( t , k ) \}$ denote the counterfactual terminal booking set in which passenger $t$ is assigned to internal option $k$ . Formally,

$$
c _ {t k} = C _ {T} \left(B _ {T}^{t,k}\right) - C _ {T} \left(B _ {T}^{-t}\right). \tag {B.6}
$$

Substituting this approximation yields the per-passenger expected-profit maximization problem stated in Eq. (13) of the main text:

$$
\max  _ {\boldsymbol {a} _ {t}} \sum_ {k \in O _ {t}} \left(f _ {e} + a _ {t k} - c _ {t k}\right) P _ {t k} \left(\boldsymbol {a} _ {t}\right). \tag {B.7}
$$

In real-time implementation, $c _ { t k }$ is replaced by its prediction $\hat { c } _ { t k }$ , which yields Eq. (14) in Section 4.2.3.

# Appendix C.

This appendix gives an interior structural characterization of the unconstrained version of the pricing problem. It is used to clarify how costs, choice probabilities, and price adjustments interact; the deployed online policy in the main text still solves the bounded problem over $\mathcal{A}$ . For simplicity, we fix a specific passenger segment $g$ and an epoch $t$ , and omit these subscripts. Let $O$ denote the set of internal options, and 0 denote the outside option. For each $k \in O$ , define

$$
\eta_ {k} = - \beta_ {g} ^ {i v t} \tau_ {k} - \beta_ {g} ^ {w a l k} d _ {k}, \quad \beta = \beta_ {g} ^ {p r i c e} > 0. \tag {C.1}
$$

Then the MNL choice probabilities in Eq. (9) and Eq. (10) can be written as

$$
P _ {k} (\boldsymbol {a}) = \frac {\exp \left(\eta_ {k} - \beta a _ {k}\right)}{\exp \left(V _ {0}\right) + \sum_ {j \in O} \exp \left(\eta_ {j} - \beta a _ {j}\right)}, k \in O, \tag {C.2}
$$

and

$$
P _ {0} (\boldsymbol {a}) = \frac {\exp \left(V _ {0}\right)}{\exp \left(V _ {0}\right) + \sum_ {j \in O} \exp \left(\eta_ {j} - \beta a _ {j}\right)}. \tag {C.3}
$$

Given a true marginal insertion-cost vector $\pmb { c } = ( c _ { k } ) _ { k \in O }$ , the unconstrained counterpart of the true-cost pricing problem in Section 4.2.3 is

$$
\max  _ {a \in \mathbb {R} ^ {| O |}} R (\boldsymbol {a}; \boldsymbol {c}) = \max  _ {a \in \mathbb {R} ^ {| O |}} \sum_ {k \in O} \left(f _ {e} + a _ {k} - c _ {k}\right) P _ {k} (\boldsymbol {a}). \tag {C.4}
$$

Let

$$
g _ {k} = f _ {e} + a _ {k} - c _ {k} \tag {C.5}
$$

denote the option-specific net profit if option $k$ is selected. Under the MNL model, the derivatives of the choice probabilities with respect to the price of option $\ell \in O$ are

$$
\frac {\partial P _ {k} (\boldsymbol {a})}{\partial a _ {\ell}} = \left\{ \begin{array}{c c} - \beta P _ {k} (\boldsymbol {a}) \left(1 - P _ {k} (\boldsymbol {a})\right), & k = \ell , \\ \beta P _ {k} (\boldsymbol {a}) P _ {\ell} (\boldsymbol {a}), & k \neq \ell . \end{array} \right. \tag {C.6}
$$

Using Eq. (C.4), the derivative of the objective with respect to $a _ { \ell }$ is

$$
\frac {\partial R (\boldsymbol {a} ; \boldsymbol {c})}{\partial a _ {\ell}} = P _ {\ell} (\boldsymbol {a}) + \sum_ {k \in O} g _ {k} \frac {\partial P _ {k} (\boldsymbol {a})}{\partial a _ {\ell}} = P _ {\ell} (\boldsymbol {a}) [ 1 + \beta (R (\boldsymbol {a}; \boldsymbol {c}) - g _ {\ell}) ]. \tag {C.7}
$$

At any interior optimum ${ { \pmb a } } ^ { * } ( { \pmb c } )$ with $P _ { \ell } ( { \pmb a } ^ { * } ) > 0$ , the first-order condition $\partial R / \partial a _ { \ell } = 0$ implies

$$
g _ {\ell} = R \left(\boldsymbol {a} ^ {*}; \boldsymbol {c}\right) + \frac {1}{\beta}, \quad \forall \ell \in O. \tag {C.8}
$$

Eq. (C.8) shows that, at any interior optimum, the option-specific net profit is equalized across all internal options. Therefore, there exists a common markup term $m ( \pmb { c } )$ such that

$$
a _ {k} ^ {*} (\boldsymbol {c}) = c _ {k} - f _ {e} + m (\boldsymbol {c}), \quad \forall k \in O. \tag {C.9}
$$

Substituting Eq. (C.9) into Eq. (C.2), we obtain

$$
P _ {k} (m; \boldsymbol {c}) = \frac {\exp \left(\eta_ {k} - \beta c _ {k} + \beta f _ {e} - \beta m\right)}{\exp \left(V _ {0}\right) + \sum_ {j \in O} \exp \left(\eta_ {j} - \beta c _ {j} + \beta f _ {e} - \beta m\right)}. \tag {C.10}
$$

Define

$$
B (\boldsymbol {c}) = \sum_ {j \in O} \exp \left(\eta_ {j} - \beta c _ {j}\right). \tag {C.11}
$$

Then the total internal-choice probability is

$$
S (m; \boldsymbol {c}) = \sum_ {k \in O} P _ {k} (m; \boldsymbol {c}) = \frac {\exp \left(\beta f _ {e} - \beta m\right) B (\boldsymbol {c})}{\exp \left(V _ {0}\right) + \exp \left(\beta f _ {e} - \beta m\right) B (\boldsymbol {c})}. \tag {C.12}
$$

Because Eq. (C.9) implies $f _ { e } + a _ { k } ^ { * } ( { \pmb c } ) - c _ { k } = m ( { \pmb c } )$ , the expected profit at any interior optimum can be rewritten as

$$
R \left(\boldsymbol {a} ^ {*} (\boldsymbol {c}); \boldsymbol {c}\right) = m (\boldsymbol {c}) S (m (\boldsymbol {c}); \boldsymbol {c}). \tag {C.13}
$$

Hence the multidimensional pricing problem reduces to

$$
\max  _ {m} m S (m; \boldsymbol {c}). \tag {C.14}
$$

Differentiating Eq. (C.12) with respect to $m$ gives

$$
\frac {\partial S (m ; \boldsymbol {c})}{\partial m} = - \beta S (m; \boldsymbol {c}) \left(1 - S (m; \boldsymbol {c})\right). \tag {C.15}
$$

Therefore, the first-order condition for Eq. (C.14) is

$$
0 = \frac {d}{d m} [ m S (m; \boldsymbol {c}) ] = S (m; \boldsymbol {c}) - \beta m S (m; \boldsymbol {c}) (1 - S (m; \boldsymbol {c})). \tag {C.16}
$$

For any interior optimum with $S ( m ; c ) > 0$ , Eq. (C.16) at $m = m ( \pmb { c } )$ yields

$$
m (\boldsymbol {c}) = \frac {1}{\beta \left(1 - S \left(m (\boldsymbol {c}); \boldsymbol {c}\right)\right)} = \frac {1}{\beta P _ {0} \left(\boldsymbol {a} ^ {*} (\boldsymbol {c})\right)}. \tag {C.17}
$$

Equivalently, using Eq. (C.12), the common markup solves the scalar fixed-point equation

$$
m (\boldsymbol {c}) = \frac {1}{\beta} \left[ 1 + \exp \left(\beta f _ {e} - V _ {0} - \beta m (\boldsymbol {c})\right) B (\boldsymbol {c}) \right]. \tag {C.18}
$$

Eq. (C.18) can be converted into a standard form to solve using the Lambert W function. We rewrite

$$
\beta m (\boldsymbol {c}) - 1 = \exp \left(\beta f _ {e} - V _ {0} - \beta m (\boldsymbol {c})\right) B (\boldsymbol {c}). \tag {C.19}
$$

Multiplying both sides by $\exp ( \beta m ( \pmb { c } ) )$ yields

$$
(\beta m (\boldsymbol {c}) - 1) \exp (\beta m (\boldsymbol {c})) = \exp (\beta f _ {e} - V _ {0}) B (\boldsymbol {c}). \tag {C.20}
$$

Now define

$$
z = \beta m (\boldsymbol {c}) - 1. \tag {C.21}
$$

Then $\beta m ( \pmb { c } ) = z + 1$ , and Eq. (C.20) becomes

$$
z \cdot \exp (z + 1) = \exp \left(\beta f _ {e} - V _ {0}\right) B (\boldsymbol {c}). \tag {C.22}
$$

Equivalently,

$$
z \cdot \exp (z) = \exp \left(\beta f _ {e} - V _ {0} - 1\right) B (\boldsymbol {c}). \tag {C.23}
$$

By the definition of the Lambert W function, $W _ { 0 } ( x ) \exp ( W _ { 0 } ( x ) ) = x$ , we obtain

$$
z = W _ {0} \left(\exp \left(\beta f _ {e} - V _ {0} - 1\right) B (\boldsymbol {c})\right). \tag {C.24}
$$

Substituting back $z = \beta m ( \pmb { c } ) - 1$ yields

$$
m (\boldsymbol {c}) = \frac {1}{\beta} \left[ 1 + W _ {0} \left(\exp \left(\beta f _ {e} - V _ {0} - 1\right) B (\boldsymbol {c})\right) \right]. \tag {C.25}
$$

Hence, the unconstrained optimal price vector is

$$
a _ {k} ^ {*} (\boldsymbol {c}) = c _ {k} - f _ {e} + \frac {1}{\beta} \left[ 1 + W _ {0} \left(\exp \left(\beta f _ {e} - V _ {0} - 1\right) B (\boldsymbol {c})\right) \right], \quad \forall k \in O. \tag {C.26}
$$

Finally, combining Eq. (C.9) and Eq. (C.17), the optimal price vector has the form

$$
a _ {k} ^ {*} (\boldsymbol {c}) = c _ {k} - f _ {e} + m (\boldsymbol {c}), \quad m (\boldsymbol {c}) = \frac {1}{\beta P _ {0} \left(\boldsymbol {a} ^ {*} (\boldsymbol {c})\right)}, \quad \forall k \in O. \tag {C.27}
$$

In the online application, however, price adjustments are limited to the feasible set $\mathcal { A } = \{ \pmb { a } _ { t } \colon -10 \leq a _ { t k } \leq 10 , \forall k \in O _ { t } \}$ outlined in Section 4.2.3. The closed-form expression above should therefore be read as an unconstrained structural guide, not as a replacement for the bounded pricing problem used by DRPO.

# References

Aflaki, A., & Zhang, Q. (2026). Is Your Price Personalized? Alleviating Customer Concerns with Inventory Availability Information. Operations Research, 74(1), 181-198.   
Akkerman, F., Dieter, P., & Mes, M. (2025). Learning dynamic selection and pricing of out-of-home deliveries. Transportation Science, 59(2), 250-278.   
Asdemir, K., Jacob, V. S., & Krishnan, R. (2009). Dynamic pricing of multiple home delivery options. European Journal of Operational Research, 196(1), 246-257.   
Bai, J., So, K. C., Tang, C. S., Chen, X., & Wang, H. (2019). Coordinating supply and demand on an ondemand service platform with impatient customers. Manufacturing & Service Operations Management, 21(3), 556-570.

Baier, M. J., Sörensen, L., & Schlüter, J. C. (2024). How successful is my DRT system? A review of different parameters to consider when developing flexible public transport systems. Transport Policy, 159, 130-142.   
Bardaka, E., Hajibabai, L., & Singh, M. P. (2020). Reimagining ride sharing: Efficient, equitable, sustainable public microtransit. IEEE Internet Computing, 24(5), 38-44.   
Bills, T. S., Twumasi-Boakye, R., Broaddus, A., & Fishelson, J. (2022). Towards transit equity in Detroit: An assessment of microtransit and its impact on employment accessibility. Transportation Research Part D: Transport and Environment, 109, 103341.   
Chandakas, E. (2020). On demand forecasting of demand-responsive paratransit services with prior reservations. Transportation Research Part C: Emerging Technologies, 120, 102817.   
Chen, Q., Wang, F., Hodge, B. M., Zhang, J., Li, Z., Shafie-Khah, M., & Catalão, J. P. (2017). Dynamic price vector formation model-based automatic demand response strategy for PV-assisted EV charging stations. IEEE Transactions on Smart Grid, 8(6), 2903-2915.   
Coutinho, F. M., van Oort, N., Christoforou, Z., Alonso-González, M. J., Cats, O., & Hoogendoorn, S. (2020). Impacts of replacing a fixed public transport line by a demand responsive transport system: Case study of a rural area in Amsterdam. Research in Transportation Economics, 83, 100910.   
Cui, S., Li, K., Yang, L., & Wang, J. (2022). Slugging: Casual carpooling for urban transit. Manufacturing & Service Operations Management, 24(5), 2516-2534.   
Currie, G., & Fournier, N. (2020). Why most DRT/Micro-Transits fail–What the survivors tell us about progress. Research in Transportation Economics, 83, 100895.   
Den Boer, A. V. (2015). Dynamic pricing and learning: Historical origins, current research, and new directions. Surveys in operations research and management science, 20(1), 1-18.   
Diao, W., Jiang, B., & Tian, L. (2023). Competition between P2P ride-sharing platforms and traditional taxis. Production and Operations Management, 32(12), 3801-3815.   
Dong, L., Kouvelis, P., & Tian, Z. (2009). Dynamic pricing and inventory control of substitute products. Manufacturing & Service Operations Management, 11(2), 317-339.   
Du, C., He, F., & Lin, X. (2025). Dynamic pricing for air cargo revenue management. Transportation Research Part E: Logistics and Transportation Review, 197, 104088.   
Elmachtoub, A. N., & Grigas, P. (2022). Smart "predict, then optimize". Management Science, 68(1), 9-26.   
Fielbaum, A., Bai, X., & Alonso-Mora, J. (2021). On-demand ridesharing with optimized pick-up and drop-off walking locations. Transportation research part C: emerging technologies, 126, 103061.   
Galiullina, A., Mutlu, N., Kinable, J., & Van Woensel, T. (2024). Demand steering in a last-mile delivery problem with home and pickup point delivery options. Transportation Science, 58(2), 454-473.   
Gerzinic, N., Gucek, M., & Cats, O. (2025). The potential of microtransit for regional commuting. European Transport Research Review, 17(1), 14.   
Ghimire, S., Bardaka, E., Monast, K., Wang, J., & Wright, W. (2024). Policy, management, and operation practices in US microtransit systems. Transport Policy, 145, 259-278.   
Haglund, N., Mladenovic, M. N., Kujala, R., Weckström, C., & Saramäki, J. (2019). Where did Kutsuplus drive us? Ex post evaluation of on-demand micro-transit pilot in the Helsinki capital region. Research in Transportation Business & Management, 32, 100390.   
Han, Z., Chen, Y., Li, H., Zhang, K., & Sun, J. (2019). Customized bus network design based on individual reservation demands. Sustainability, 11(19), 5535.   
Jang, M., Lee, S., Kim, J., & Kim, J. (2025). Dynamic minimum service level of demand–responsive transit: a prospect theory approach. Sustainability, 17(7), 3171.

Kaddoura, I., Leich, G., & Nagel, K. (2020). The impact of pricing and service area design on the modal shift towards demand responsive transit. Procedia Computer Science, 170, 807-812.   
Kayikci, Y., Demir, S., Mangla, S. K., Subramanian, N., & Koc, B. (2022). Data-driven optimal dynamic pricing strategy for reducing perishable food waste at retailers. Journal of cleaner production, 344, 131068.   
Kersting, M., Kallbach, F., & Schlüter, J. C. (2021). For the young and old alike--An analysis of the door-to-door DRT system EcoBus in rural Germany. Journal of Transport Geography, 96, 103173.   
Knierim, L., & Schlüter, J. C. (2021). The attitude of potentially less mobile people towards demand responsive transport in a rural area in central Germany. Journal of Transport Geography, 96, 103202.   
Kostic, B., Loft, M. P., Rodrigues, F., & Borysov, S. S. (2021). Deep survival modelling for shared mobility. Transportation Research Part C: Emerging Technologies, 128, 103213.   
Lee, E., Cen, X., Lo, H. K., & Ng, K. F. (2021). Designing zonal-based flexible bus services under stochastic demand. Transportation Science, 55(6), 1280-1299.   
Lei, D., Xu, M., & Wang, S. (2024). A conditional diffusion model for probabilistic estimation of traffic states at sensor-free locations. Transportation Research Part C: Emerging Technologies, 166, 104798.   
Lei, D., Xu, M., & Wang, S. (2025). A deep multimodal network for multi-task trajectory prediction. Information Fusion, 113, 102597.   
Li, Y., Li, X., & Zhang, S. (2021). Optimal pricing of customized bus services and ride-sharing based on a competitive game model. Omega, 103, 102413.   
Li, Y., Sun, H., Chang, X., Lv, Y., & Gao, K. Towards Smarter On-Demand Ride-Sharing: Balancing Efficiency and Flexibility Via Delayed Matching and Walking. Available at SSRN 5346078.   
Liu, T., & Ceder, A. A. (2015). Analysis of a new public-transport-service concept: Customized bus in China. Transport Policy, 39, 63-76.   
Liu, X., Chen, X., Potoglou, D., Tian, M., & Fu, Y. (2023). Travel impedance, the built environment, and customized-bus ridership: A stop-to-stop level analysis. Transportation Research Part D: Transport and Environment, 122, 103889.   
Lyu, Y., Chow, C. Y., Lee, V. C., Ng, J. K., Li, Y., & Zeng, J. (2019). CB-Planner: A bus line planning framework for customized bus systems. Transportation Research Part C: Emerging Technologies, 101, 233-253.   
Mayaud, J. (2025). On the role of microtransit in shaping new mobility patterns. Travel Behaviour and Society, 41, 101065.   
Miah, M. M., Naz, F., Hyun, K. K., Mattingly, S. P., Cronley, C., & Fields, N. (2020). Barriers and opportunities for paratransit users to adopt on-demand micro transit. Research in transportation economics, 84, 101001.   
Miller, P., de Barros, A. G., Kattan, L., & Wirasinghe, S. C. (2016). Analyzing the sustainability performance of public transit. Transportation Research Part D: Transport and Environment, 44, 177- 198.   
Naumov, S., & Keith, D. (2023). Optimizing the economic and environmental benefits of ride‐hailing and pooling. Production and Operations Management, 32(3), 904-929.   
Pellegrini, A., & Fielbaum, A. (2025). Are users ready to accept fully flexible walking in on-demand mobility?. Transportation Research Part C: Emerging Technologies, 178, 105210.   
Quadrifoglio, L., Dessouky, M. M., & Ordóñez, F. (2008). A simulation study of demand responsive transit system design. Transportation Research Part A: Policy and Practice, 42(4), 718-737.

Rath, S., Liu, B., Yoon, G., & Chow, J. Y. (2023). Microtransit deployment portfolio management using simulation-based scenario data upscaling. Transportation Research Part A: Policy and Practice, 169, 103584.   
Riggs, W., & Pande, A. (2022). On-demand microtransit and paratransit service using autonomous vehicles: Gaps and opportunities in accessibility policy. Transport Policy, 127, 171-178.   
Saharan, S., Bawa, S., & Kumar, N. (2020). Dynamic pricing techniques for Intelligent Transportation System in smart cities: A systematic review. Computer Communications, 150, 603-625.   
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.   
Shang, H., Chang, Y., Huang, H., & Zhao, F. (2022). Integration of conventional and customized bus services: An empirical study in Beijing. Physica A: Statistical Mechanics and its Applications, 605, 127971.   
Shu, W., & Li, Y. (2022). A novel demand-responsive customized bus based on improved ant colony optimization and clustering algorithms. IEEE Transactions on Intelligent Transportation Systems, 24(8), 8492-8506.   
Sörensen, L., Bossert, A., Jokinen, J. P., & Schlüter, J. (2021). How much flexibility does rural public transport need?–Implications from a fully flexible DRT system. Transport Policy, 100, 5-20.   
Sultana, Z., Mishra, S., Cherry, C. R., Golias, M. M., & Jeffers, S. T. (2018). Modeling frequency of rural demand response transit trips. Transportation Research Part A: Policy and Practice, 118, 494-505.   
Tong, L. C., Zhou, L., Liu, J., & Zhou, X. (2017). Customized bus service design for jointly optimizing passenger-to-vehicle assignment and vehicle routing. Transportation Research Part C: Emerging Technologies, 85, 451-475.   
Wang, J., Liu, K., Yamamoto, T., Wang, D., & Lu, G. (2023). Built environment as a precondition for demand-responsive transit (DRT) system survival: Evidence from an empirical study. Travel Behaviour and Society, 30, 271-280.   
Wang, L., Zeng, L., Ma, W., & Guo, Y. (2021). Integrating passenger incentives to optimize routing for demand-responsive customized bus systems. Ieee Access, 9, 21507-21521.   
Wu, B., Zuo, X., Chen, G., Ai, G., & Wan, X. (2024). Multi-agent deep reinforcement learning based realtime planning approach for responsive customized bus routes. Computers & Industrial Engineering, 188, 109840.   
Wu, W., Zhang, Z., Lu, K., & Ren, J. (2025). Dynamic demand-responsive transit scheduling with timedependent travel times: A joint supply and demand management approach. Transportation Research Part E: Logistics and Transportation Review, 202, 104232.   
Xia, D., Zheng, L., Cai, X., Liu, W., & Sun, D. (2022). Urban customized bus design for private car commuters. IEEE Internet of Things Journal, 9(21), 21723-21735.   
Yan, J., Martin, S., & Taylor, S. J. (2025). Trading flexibility for adoption: From dynamic to static walking in ride-sharing. Management Science, 71(7), 5875-5892.   
Yang, X., Strauss, A. K., Currie, C. S., & Eglese, R. (2016). Choice-based demand management and vehicle routing in e-fulfillment. Transportation science, 50(2), 473-488.   
Yu, C., Lin, H., Chen, Y., Yang, C., Yin, A., & Yuan, Q. (2024). Creating most needed customized bus services: A collaborative analysis of user-route dynamics. Transportation Research Part D: Transport and Environment, 133, 104312.   
Yu, C., Lin, H., Chen, Y., Yang, C., Yin, A., & Yuan, Q. (2024). Creating most needed customized bus services: A collaborative analysis of user-route dynamics. Transportation Research Part D: Transport and Environment, 133, 104312.

Zhang, W., Jacquillat, A., Wang, K., & Wang, S. (2023). Routing optimization with vehicle–customer coordination. Management Science, 69(11), 6876-6897.   
Zhang, Y., Luo, S., Zu, A., Ji, H., Kang, L., & Shao, C. (2025). Optimal design of fixed-route and demandresponsive transit with a dynamic stop strategy. Transportation Research Part A: Policy and Practice, 199, 104581.   
Zhao, Z., & Lee, C. K. (2021). Dynamic pricing for EV charging stations: A deep reinforcement learning approach. IEEE Transactions on Transportation Electrification, 8(2), 2456-2468.   
Zheng, Y., Li, W., Qiu, F., & Wei, H. (2019). The benefits of introducing meeting points into flex-route transit services. Transportation Research Part C: Emerging Technologies, 106, 98-112.
