

template<class INFERENCE>
class VisitorInterface{

public:
	void begin(INFERENCE & inf);
	void end(INFERENCE & inf);
	bool operator()(INFERENCE & inf)
}