# Latent Symbolic Planning

Classical planning is an invaluable approach to solving MDP-like problems, especially when we require completeness guarantees. However, when states or observations become high-dimensional (e.g., image data) or continuous, many classical planning methods (e.g. PDDL 1.0) become bedeviled by the curse of dimensionality — often becoming intractable.  
	
In this project, I question whether we can compress high-dimensional / continuous representations into domains where classical planning is feasible and propose a methodology to do just that. I plan to achieve consistent solving of the LunarLander environment and compare performance of this method—measured in terms of environment interactions, time before reaching the goal, and total rewards/returns achieved—against traditional RL baselines.

# Methodology



#### LLMs were used for
 - Docstrings
 - Test-Driven Development (TDD)
 - Code evaluation and bug identification
 - `analyze_k_clusters` and clustering code
 - plotters
 - PDDL converter
 - Some of acting loop
