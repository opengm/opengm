


template<class GM>
class HighLevelFusionMover{

	struct Parameter{


		/*
			// WORKFLOW FOR SECOND ORDER GRAPHICAL MODELS
			std::string wf="(Qpbo-I)-(LF2)"
			std::string wf="(Qpbo-P)-(LF2)"
		*/
		std::string secondOrderWorkflow_;

		/*
			// WORKFLOW FOR HIGHER ORDER GRAPHICAL MODELS

			// plain opt  workflow 
			std::stringt wf = "(ad3)"
			std::stringt wf = "(cplex)"
			std::stringt wf = "(astar)"
			std::stringt wf = "(bf)"

			// size controll workflow 
			
			std::string wf = "( [(ad3){lpVarRange}   ) "

		*/
		std::string highOrderWorkflow_;
	};
}