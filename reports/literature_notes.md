# Literature Notes — Multi-Model Routing (Week 2)

Paper: RouteLLM
Core routing signal used:
	Probability of strong model outperforming weak model on a given query (win probability) (Page 3, Win Prediction Model)
How decision is made:
	Trained router model estimated P(Strong wins | Query). If above threshold, query routed to strong model. Otherwise, weak model. (Page 3, Equation (2) )
Confidence method used:
	Learned win-prediction model gives a probability score using preference data and sigmoid/logistic formulation. Acts as a confidence style score for strong vs weak model performance.
Difficulty estimation method:
	Difficulty inferred indirectly from preference data. Router learns from human preference comparisons between strong and weak model outputs and predicts what queries require the stronger model based off of that. (Pg. 3, 4-5) Look for "learning routers from preference data"
Metrics used:
	Percent calls to strong model (cost rate)
	Average response quality
	Performance Gap Recovered (PGR)
	Average Performance (APGR)
	Call-Performance Threshold (CPT)
	Cost savings ratio
	Page 4 - Section 3.2 Metrics
	Page 2 - Figure 1 
Why it matters for my router:
	Direct framework for confidence based routing between small / large models using probability threshold, which matches my planned routing strategy using escalation based on confidence scores
	Page 1-2 - motivation and cost/quality routing goal - "Powerful models offer better results but are expensive" "each user query through a router", 
	Page 1-2 - Routing simpler queries to cheaper models
	"router can direct simpler queries"
	Page 3 - binary routing with threshold
	"the routing decision is defined as"
What I can replicate this semester:
	Routing based on thresholds between s/m/l models with predicted confidence scores, cost vs accuracy evaluation curves, routing logs, and escalation decisions using a learned or heuristic confidence predictor 
	Page 3 - Binary routing with threshold
	"We introduce a principled framework for learning a binary routing function"
	Page 3 - Learned confidence / WP model
	"This model estimates the probability that a strong model"
	Page 6 - Feasible router implementations
	"We now discuss several methods to define the win prediction model"
	Page 6-7 - Practical routing implementations (BERT, MF, Similarity routing)
	Shows threshold routing and learned predictor is feasible
	"We explore using a standard text classification method"
	"Drawing inspiration from matrix factorization models...learn from preference data"


Paper: WandB Router Guide
Core routing signal used:
	Predict likelihood that query req. strong or weak model, based on past performances and alignment score
	Page 2 - "dynamically directs queries to the most appropriate"
How decision is made:
	The router predicts whether weak model will handle query sufficiently. If not, routed to stronger model. Implemented as binary classification with a threshold on routers confidence score. 
	Page 3 - "predict the likelihood of each model’s success"
Confidence method used:
	Trained binary classifier put a logit through a sigmoid function to produce a probability score. Represents confidence that stronger model is needed, compared against threshold alpha to make decisions on routing.
	Page 6 - "This code trains a binary classifier that.."
	Page 17 - "we identify confidence thresholds (alpha) which will results"
Difficulty estimation method:
	Inferred from model performance alignemtn scores. Queries that weak models perform poorly on (below some threshold) treated as difficult and routed to strong model. No explicit difficulty labels.
	Page 4 - "Responses rated 4 or higher are considered sufficient..."
Metrics used:
	Performance Gap Recovered (PGR)
	Call-Performance Threshold (CPT)
	Accuracy vs % calls to strong model
	Cost / performance trade-off curves
	Training and validation accuracy
	Page 12 - "We use two key metrics"
	Page 13 - "Performance / cost trade-off charts"
Why it matters for my router:
	It's a concrete, end-to-end implementation of a confidence based routing system, including dataset construction, classifier training, threshold tuning, and evaluation. Validates my idea of escalation based routing with confidence thresholds and cost vs accuracy analysis.
	Page 1 - "balancing performance and cost"
	Page 2 - Routing simpler queries to cheaper models "dynamically directs queries to the most appropriate"
	Page 17 - tuning routing w confidence alpha "we identify confidence thresholds (alpha) which will result in around 50%"
What I can replicate this semester:
	Trained classifier based router with labeled performance data, routing queries using sigmoid confidence score and threshold, logging of routing decisions, cost vs accuracy trade-off plots, computing PGR and CPT metrics without large-scale infrastructure.
	Page 6 - Binary classifier for routing "code trains a binary classifier"
	Page 13-17 - Evaluations across bins and threshold tuning
	"By sorting the model's confidence scores"
	"This approach allows for precise identification of the optimal balance"
	Page 20-22 - Routing decisions logic with alpha
	"function evaluates whether the response matches the expected outcome" 
	"alpha value used in the routing decision was previously obtained"
	

Paper: Mendoza Model Selection
Core routing signal used:
	Different models react differently to different requests, so the system has to pick the right model accordingly per request.
	Page 2 - “...model selectors and schedulers employ model selection and scheduling…to assign queries to models on workers that can satisfy their SLOs.”
How decision is made:
	Model selection decision made for each request before inference execution.
	Page 3 - “Queries from inference applications arrive at a central queue…assigned to models on workers by a model selector and scheduler.”
Confidence method used:
	Does not use probabilistic confidence score, routing is a model selection decision through model selector and scheduler instead of prediction of confidence.
	Page 3 - “Queries from inference applications arrive at a central queue…assigned to models on workers by a model selector and scheduler.”
Difficulty estimation method:
	Not explicitly labeled, requests just differ in characteristics which effect model behavior.
	Page 1 - "MS&S is challenged by varying query load...and...query inter arrival patterns..." 
	Page 3 - "Shifts in query load impact the set of models capable of satisfying an application's latency SLO." 
Metrics used:
	Evaluated based on system level effects of model choice.
	Page 11 - "We compare...for each classification task and work-load...with respect to their observed Latency SLO Violation Rate and Accuracy Per Satisifed Query,"
Why it matters for my router:
	Shows routing as an essential upstream decision, selecting which model to handel a request. Shows that routing decisions directly affect system performance downstream, which justifies why I need intelligent routing logic.
	Page 3 - “Queries from inference applications arrive at a central queue…assigned to models on workers by a model selector and scheduler.”
		Shows that it happens before execution, assigned to models with a model selector
	Page 11 - "We compare...for each classification task and work-load...with respect to their observed Latency SLO Violation Rate and Accuracy Per Satisifed Query,"
		Shows that evaluation compares outcomes, and these outcomes do depend on model choice, so routing impacts performance downstream.
What I can replicate this semester:
	I can replicate the routing decision, aka selecting which model to handle a request without implementing a latency or cost optimization or enforcing SLOs.
	Page 3 - “Queries from inference applications arrive at a central queue…assigned to models on workers by a model selector and scheduler.”
		Shows that model selection exists per request.


Paper: Dynamic Neural Networks Survey
Core routing signal used:
	Dynamic neural networks adapt their computation paths based on input, so different inputs activate different parts of the model.
	Page 1 - "dynamic networks can adapt their structures or parameters to different inputs"
How decision is made:
	The network decides during inference which path or components of computation to execute for a given input.
	Page 1 - "selectively activating model components...conditioned on the input."
Confidence method used:
	Some dynamic netwworks do use signals of confidence or uncertainty to decide to continue or stop computation early.
	Page 3 - "adaptive early exiting is typically achieve according to confidence-based criteria...or learned functions"
	Page 3 - "samples should be output at certain early exits without executing deeper layers"
Difficulty estimation method:
	Difficulty not labeled explicitly, but harder inputs trigger more computation.
	Page 3 - "Dynamic architectures not only save redundant computation for canonical ("easy") samples, but preserve representation power...[for]non-canonical ("hard") samples."
Metrics used:
	Evaluated on accuracy vs efficiency of computation.
	Page 12 - "A trade-off between accuracy and efficiency is controlled by manipulating the thresholds...usually tuned on a validation dataset."
Why it matters for my router:
	Paper shows that computation based on input has been done and is well established, supporting design of model routers. 
	Page 3 - "Dynamic architecutres...save redundant computation...preserve...power...remakrable advantages in efficiency compared to the acceleration techniques for static models...handle...inputs with identical computation...fail to reduce...computational redundancy"
		Shows a reason for dynamic computation and why static can be bad.
What I can replicate this semester:
	I can replicate the idea of routing decisions based on input, but without modifying internal network structures.
	"Page 1 - "dynamic networks can adapt their structures or parameters to different inputs"

Paper: BranchyNet
Core routing signal used:
	Routing based on the confidence of predictions allowing early exits when confidence is high.
	Page 1 - "The architecture allows prediction results for a large portion of test samples to exit the network early via these branches when samples can already be inferred with high confidence."
How decision is made:
	Every exit point is evaluated in sequence to decide whether exiting early or continuing is better based on a certain confidence threshold.
	Page 2 - "If the entropy of a test sample is below a learned threshold value, meaning that the classifier is confident in the prediction, the sample exits the network with the prediction result at this exit point, and is not processed by the higher network layers.”
Confidence method used:
	Uses equation called entropy of softmax output as a confidence measure to see if exiting early is a good idea.
	Page 2 - "At each exit point, BranchyNet uses the entropy of a classification result (e.g., by softmax) as a measure of confidence in the prediction.”
Difficulty estimation method:
	Not labeled explicitly, just that easier exits early and vice verse for harder.
	Page 1 - “For more difficult samples, which are expected less frequently, BranchyNet will use further or all network layers to provide the best likelihood of correct prediction.”
Metrics used:
	Uses accuracy and inference time / efficiency.
	Page 1 - "..show that it can both imrpove accuracy and significantly reduce the inference time..."
	Page 4 - "..outperforms the original..."
Why it matters for my router:
	Shows that routing based on confidence can reduce computation that's not necessary while keeping accuracy, so it further enforces use of confidence based routing strategies.
	Page 1 - "By exiting...with prediction at earlier stages....significantly reduces the runtime and energy use..."
	Page 1 - " it can both improve accuracy and significantly reduce the inference time of the network"
What I can replicate this semester:
	I can replicate using routing  with confidence threshold based escalation to stop or continue computation but not implement the modifications of BranchyNet lke early exit branches.
	Page 2 - "If the entropy value is above the threshold, then the classifier at this exit point is deemed not confident, and the sample continues to the next exit point in the network."

Paper: MoE Routing (skim)
Core routing signal used:
	Routing allocates experts to each input based off of the confidence level of each expert for that input. Only selected experts are used and their outputs combined. Experts are sub neural networks within a larger model.
	Page 1 - "Our method dynamically allocates experts based on the confidence level in expert selection for each input."
	Page 3 - "We regard that P in Equation 2 reflects the confidence level of input x in selecting different experts."
How decision is made:
	Experts are selected by sorting probabilities per experts and activating experts until the added probability between them is higher than the threshold given.
	Page 2 - “Each token selects experts with higher routing probabilities until the cumulative probability exceeds the threshold.”
	Page 3 - "We find the samllest set of experts...whose cumulative probability exceeds the threshold..."
Confidence method used:
	Softmax probability assigned to every expert is the confidence score, for the experts suitability to each input.
	Page 3 - "...represents how confident the mdoel is that the...[i]th expert can adequately handle input [x]..."
Difficulty estimation method:
	Not labeled explicitly, but inferred from amount of uncertainty in selecting experts and how many experts need to be activated.
	Page 1- "The more challenging input…might need more parameters to solve."
	Page 2 - "Dispatching experts equally across inputs could lead to computational waste on simpler tasks and insufficient computational resources for more difficult ones."
	Page 5 - "Solving the BBH[hard] task requires activating an average of 1.87 experts, more than the number needed for other tasks."
Metrics used:
	Uses task accuracy and number of experts activated as a way to calculate computational efficiency.
	Page 5 - "Table 1: Performance on downstream tasks."
	Page 6 - "We calculate the average number of experts activated by the model across different downstream tasks."
Why it matters for my router:
	Shows that routing on confidence and inferred difficulty can change its computation per input, which strengthens the idea that routing should depend on complexity of input instead of being fixed.
	Page 2 - "The TopK Routing...overlooks the different difficulties of inputs...could lead to computational waste on simpler tasks and insufficient computational resources for more difficult ones."
	Page 7 - "Our model’s use of more experts on BBH [hard] tasks implies that our method can dynamically monitor task difficulty and apply more parameters to tackle more challenging tasks."
What I can replicate this semester:
	I can replicate the logic of the routing as in using probability thresholds to see whether to escalate to stronger models but not implement a full MoE architecture.
	Page 3 "If the highest probability is not large enough, we need to add more experts to increase the reliability of processing..."


