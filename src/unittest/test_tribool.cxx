#include <stdlib.h>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <map>
#include <stdexcept>

#include "opengm/utilities/tribool.hxx"

void test(bool testResult,std::string testName,std::string reasonFail=std::string(""))
{
	if(testResult==true)
	{
		std::cout<<"[o]"<<" PASSED: "<<testName<<std::endl;
	}
	else
	{
		std::cout<<"[-]"<<" FAILED: "<<testName<<std::endl;
		if(reasonFail.size()!=0)
		{
			std::cout<<"	Reason: "<<reasonFail<<std::endl;
		}
		throw std::logic_error("test failed.");
	}
}


void testTribool()
{
	{
		opengm::Tribool a(0);
		test((a==false) ,"constructor");
		opengm::Tribool b(false);
		test((b==false) ,"constructor");
		a=true;
		test((a==true) ,"compare");
		test((a==opengm::Tribool::True) ,"compare");
		bool t=false;
		if(a)
		{
			t=true;
		}
		test((t==true) ,"compare");

		if(!a)
		{
			t=false;
		}
		test((t==true) ,"compare");
	}
	{
		opengm::Tribool a(1);
		test(a==true ,"constructor");
		a=false;
		test((a==false) ,"compare");
		test((a==opengm::Tribool::False) ,"compare");
		bool t=false;
		if(a)
		{
			t=true;
		}
		test((t==false) ,"compare");
		if(!a)
		{
			t=true;
		}
		test((t==true) ,"compare");
	}
	{
		opengm::Tribool a(opengm::Tribool::Maybe);
		test(a!=true && a!=false ,"constructor");
		test((a==-1) ,"compare");
		bool t=false;
		if(a)
		{
			t=true;
		}
		test((t==false) ,"compare");
		if(!a)
		{
			t=true;
		}
		test((t==false) ,"compare");
		test( (a.maybe()) ,"compare");

	}
}



int main(int argc, char** argv) {
	testTribool();
    return (EXIT_SUCCESS);
}


