#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <limits>
#include <algorithm>


#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/explicit_function.hxx>

int main(int argc, const char* argv[] ) {
   if(argc != 3 && argc != 4) {
      std::cerr << "Two input arguments required" << std::endl;
      return 1;
   }

   typedef double ValueType;
   typedef size_t IndexType;
   typedef size_t LabelType;
   typedef opengm::Adder OperatorType;
   typedef opengm::Minimizer AccumulatorType;
   typedef opengm::DiscreteSpace<IndexType, LabelType> SpaceType;

   // Set functions for graphical model
   typedef opengm::meta::TypeListGenerator<
      opengm::ExplicitFunction<ValueType, IndexType, LabelType>
   >::type FunctionTypeList;

   typedef opengm::GraphicalModel<
      ValueType,
      OperatorType,
      FunctionTypeList,
      SpaceType
   > GmType;

   GmType gm;
   std::string opengmfile   = argv[2];
   std::string matchingfile = argv[1];
   double lambda            = 1;
   if(argc==4) lambda = atof(argv[3]);
   std::cout << lambda << std::endl;

   ValueType   infty        = 1000000;

   // open matching file
   std::ifstream mymatchingfile(matchingfile.c_str());
   if(!mymatchingfile.is_open()) {
      std::cerr << "Could not open file " << matchingfile << std::endl;
      return 1;
   }

   // temp storage for current line of uai file
   std::string currentLine; 
   char command;
   size_t pointsLeft;
   size_t pointsRight;
   size_t A;
   size_t E; 

   mymatchingfile >> command;
   if(command=='c'){
      std::string line;
      std::getline(mymatchingfile, line);
      mymatchingfile >> command;
   }
   if(command!='p'){
      std::cerr << "Missing Model info (p)!" << std::endl;
      return 1;
   }
   mymatchingfile >> pointsLeft;
   mymatchingfile >> pointsRight;
   mymatchingfile >> A;
   mymatchingfile >> E;
   std::vector<std::pair<size_t, size_t> > assignments(A); 
   std::vector<double >                    assignmentsV(A);
   std::vector<std::pair<size_t, size_t> > assignmentsVar(A);
   std::vector<std::pair<size_t, size_t> > edges(E); 
   std::vector<double >                    edgesV(E);

   std::cout << "Problem has "<<pointsLeft<<" and "<<pointsRight<<" points to match. ("<<A<<","<<E<<")"<<std::endl;

   int countA        = 0; 
   int countE        = 0;
   size_t countL     = 0;
   int countKeyLeft  = 0;
   int countKeyRight = 0;
   int flag          = 0;
   int id            = -1;
   std::map<size_t,size_t> keysLeft;
   std::map<size_t,size_t> keysRight;
   std::vector<size_t>     keys;
   std::vector<std::vector<size_t> > labels;


   while(!mymatchingfile.eof()) {
      mymatchingfile >> command;
      if(command =='a'){
         ++countA;
         int aID, pL, pR;
         ValueType v;
         mymatchingfile >> aID;
         mymatchingfile >> pL;  
         mymatchingfile >> pR;
         assignments[aID] = std::pair<size_t,size_t>(pL,pR);
         mymatchingfile >> v;
         assignmentsV[aID] = v; 

         if(flag==0){ // Left -> Right
            if(pL>id){
               id=pL;
               countL=0;
               keysLeft[pL] = countKeyLeft;
               keys.push_back(pL);
               ++countKeyLeft;
            }
            if(pL<id){
               flag=1;
               id=-1;
            }
            else{
               assignmentsVar[aID] =  std::pair<size_t,size_t>(countKeyLeft-1, countL++);
            }

         }
         if(flag==1){ // Right -> Left
           
            if(pR>id){
               id=pR;
               countL=0; 
               keysRight[pR] = countKeyLeft+countKeyRight;
               keys.push_back(pR);
               ++countKeyRight;
            }
           
            assignmentsVar[aID] =  std::pair<size_t,size_t>(countKeyLeft+countKeyRight-1, countL++);
         }
      }
      if(command =='e'){++countE;
         size_t aID1, aID2;
         double v;
         mymatchingfile >> aID1;
         mymatchingfile >> aID2;
         mymatchingfile >> v;  
         edges[countE-1] = std::pair<size_t,size_t>(aID1,aID2);
         edgesV[countE-1] = lambda*v; 
      }
   }
   std::cout << "Left keyfeatures: " << countKeyLeft << " Right keyfeatures: " << countKeyRight <<std::endl;
   size_t numVar =  countKeyLeft+  countKeyRight;
   std::vector<std::vector< size_t> >  Adj(numVar,std::vector<size_t>(numVar,0)); 
 
   labels.resize(numVar);
   // add original labels
   for (size_t i=0; i< assignments.size(); ++i){
      const size_t var = assignmentsVar[i].first;
      if(var<countKeyLeft)
         labels[var].push_back(assignments[i].second);
      else
         labels[var].push_back(assignments[i].first);
   } 
   // add reverse induced labels
   std::vector<std::vector<size_t> > duplicates;
   for (size_t i=0; i< assignments.size(); ++i){
      if(assignmentsVar[i].first<countKeyLeft){
         //check if right feature is keyfeature
         std::map<size_t,size_t>::iterator it = keysRight.find(assignments[i].second);
         if(it !=  keysRight.end()){
            const size_t var  = it->second;
            const size_t feat = assignments[i].first;
            //check if label exists
            bool collision = false;
            bool label     = 0;
            for(size_t j=0; j<labels[var].size();++j)
               if(labels[var][j] == feat){
                  collision = true;
                  label     = j;
               }
            if(!collision){
               labels[var].push_back(feat); 
               std::vector<size_t> c(4);
               c[2]=var;
               c[3]=labels[var].size()-1;
               c[0]=assignmentsVar[i].first;
               c[1]=assignmentsVar[i].second;
               duplicates.push_back(c);
               Adj[c[0]][c[2]]=1;
               // std::cout << var << " --> " << feat <<std::endl;
            }
            else{
               //std::cout << "collision: ( "<< var<<","<< feat<<") -- ("<<assignmentsVar[i].first<<","<<assignments[i].second<<")" <<std::endl;
               //labels[var].push_back(feat);
               std::vector<size_t> c(4);
               c[2]=var;
               c[3]=label;
               c[0]=assignmentsVar[i].first;
               c[1]=assignmentsVar[i].second;
               duplicates.push_back(c);
               Adj[c[0]][c[2]]=1;
            }
         }
      }else{
         //check if left feature is keyfeature
         std::map<size_t,size_t>::iterator it = keysLeft.find(assignments[i].first); 
         if(it !=  keysLeft.end()){ 
            //check if label exists
            const size_t var  = it->second;
            const size_t feat = assignments[i].second;
            bool collision= false; 
            bool label     = 0;
            for(size_t j=0; j<labels[ it->second].size();++j)
               if(labels[var][j] == feat){
                  collision = true;   
                  label     = j;
               }
            if(!collision){ 
               labels[var].push_back(feat);
               std::vector<size_t> c(4);
               c[0]=var;
               c[1]=labels[var].size()-1;
               c[2]=assignmentsVar[i].first;
               c[3]=assignmentsVar[i].second;
               duplicates.push_back(c);
               Adj[c[0]][c[2]]=1;
               //std::cout << var<< " --> " << feat <<std::endl;
            }else{
               //std::cout << "collision: ( "<< var <<","<< feat <<") -- ("<<assignmentsVar[i].first<<","<<assignments[i].first<<")" <<std::endl; 
               //labels[var].push_back(feat);
               std::vector<size_t> c(4);
               c[0]=var;
               c[1]=label;
               c[2]=assignmentsVar[i].first;
               c[3]=assignmentsVar[i].second;
               duplicates.push_back(c);
               Adj[c[0]][c[2]]=1;
            }
         }
      }
   } 

   for(size_t var=0; var<labels.size();++var){
      if (var<countKeyLeft){      
         for (size_t l=0; l<labels[var].size(); ++l){
            std::map<size_t,size_t>::iterator it = keysRight.find(labels[var][l]);
            if(it !=  keysRight.end()){
               const size_t var2  = it->second;
               size_t l2     = 0;
               for(l2 = 0; l2<labels[var2].size();++l2){
                  if(labels[var2][l2] == keys[var] ){
                     break;
                  }
               }
               std::vector<size_t> c(4);
               c[0]=var;
               c[1]=l;
               c[2]=var2;
               c[3]=l2;
               duplicates.push_back(c);
               Adj[var][var2]=1;
            }
         }     
      }else{
         for (size_t l=0; l<labels[var].size(); ++l){
            std::map<size_t,size_t>::iterator it = keysLeft.find(labels[var][l]);
            if(it !=  keysLeft.end()){
               const size_t var2  = it->second;
               size_t l2     = 0;
               for(l2 = 0; l2<labels[var2].size();++l2){
                  if(labels[var2][l2] == keys[var]){
                     break;
                  }
               }
               std::vector<size_t> c(4);
               c[2]=var;
               c[3]=l;
               c[0]=var2;
               c[1]=l2;
               duplicates.push_back(c);
               Adj[var2][var]=1;
            }
         }     
      }
   }



   // add hidden labels
   for(size_t i=0; i<labels.size();++i){
      labels[i].push_back(std::numeric_limits<size_t>::max());
   }
   for(size_t i=0; i<edges.size(); ++i){
      //std::cout << assignmentsVar[edges[i].first].first << " " << assignmentsVar[edges[i].second].first <<std::endl;
      Adj[assignmentsVar[edges[i].first].first][assignmentsVar[edges[i].second].first] = 1;
   }

   for(size_t i=0; i<countKeyLeft; ++i)
      std::cout << labels[i].size() << " ";
   std::cout << std::endl; 
   for(size_t i=0; i<countKeyRight; ++i)
      std::cout << labels[i+countKeyLeft].size() << " ";
   std::cout << std::endl;

   if (A!=countA)
      std::cout << "Wrong number of Assignments "<<  A <<" != " << countA <<std::endl;
   if (E!=countE)
      std::cout << "Wrong number of Edges "<<  E <<" != " << countE <<std::endl;
 

/*
   size_t count=0;
   for (size_t i=0; i<numVar; ++i){
      for (size_t j=0; j<numVar; ++j){
         std::cout <<Adj[i][j] <<" ";
         count +=Adj[i][j];
      }
      std::cout <<std::endl;
   }
   std::cout<<count <<std::endl; 
*/


   std::cout << "Build model ..."<<std::endl; 
   std::vector<opengm::ExplicitFunction<ValueType, IndexType, LabelType> >    unaries(countKeyLeft+countKeyRight);
   for(IndexType i=0; i<numVar; ++i){
      LabelType numLabels = labels[i].size();
      gm.addVariable(numLabels);
      unaries[i] = opengm::ExplicitFunction<ValueType, IndexType, LabelType>(&numLabels,&numLabels+1);
   }

   std::cout << "Fill unaries ..."<<std::endl; 
   for(size_t i=0; i<assignmentsV.size() ;++i){
      unaries[assignmentsVar[i].first]( assignmentsVar[i].second ) = assignmentsV[i]; 
   }

   std::cout << "Add unaries ..."<<std::endl;
   for(IndexType var=0; var<gm.numberOfVariables(); ++var){
      GmType::FunctionIdentifier currentFunctionIndex = gm.addFunction(unaries[var]);
      gm.addFactor(currentFunctionIndex, &var, &var+1);
   }

   std::cout << "Create pairwise terms ..."<<std::endl; 
   std::vector< std::vector< opengm::ExplicitFunction<ValueType, IndexType, LabelType> > >   pairs(countKeyLeft+countKeyRight, std::vector< opengm::ExplicitFunction<ValueType, IndexType, LabelType> >(countKeyLeft+countKeyRight)); 
   for (size_t i=0; i<countKeyLeft+countKeyRight; ++i){
      for (size_t j=0; j<countKeyLeft+countKeyRight; ++j){
         if(Adj[i][j]==1){
            LabelType nL[2];
            nL[0] = gm.numberOfLabels(i);
            nL[1] = gm.numberOfLabels(j);
            pairs[i][j] = opengm::ExplicitFunction<ValueType, IndexType, LabelType>(nL,nL+2);
         }
      }
   }
   std::cout << "Fill pairwise terms ..."<<std::endl; 
   for(size_t i=0;i<edges.size();++i){
      size_t a1 = edges[i].first;
      size_t a2 = edges[i].second;
      size_t var1 = assignmentsVar[a1].first;
      size_t var2 = assignmentsVar[a2].first;
      size_t l1 = assignmentsVar[a1].second;
      size_t l2 = assignmentsVar[a2].second;

      if(var1<var2){
         pairs[var1][var2](l1,l2)=edgesV[i];
      }
      else{
         pairs[var2][var1](l2,l1)=edgesV[i];
      }
   } 

   for(size_t i=0; i<duplicates.size(); ++i){
      const size_t var1 = duplicates[i][0];
      const size_t var2 = duplicates[i][2]; 
      const size_t label1 = duplicates[i][1];
      const size_t label2 = duplicates[i][3];

      //std::cout << var1 << " " << label1 << " - "<< var2 << " " << label2 <<std::endl;
      for(LabelType l1=0; l1<gm.numberOfLabels(var1); ++l1)
         if(l1 != label1)
            pairs[var1][var2](l1,label2) = 100000.0;
      for(LabelType l2=0; l2<gm.numberOfLabels(var2); ++l2)
         if(l2 != label2)
            pairs[var1][var2](label1,l2) = 100000.0;
   }


   std::cout << "Add constraints to pairwise terms to enforce 1 to 1 match ..."<<std::endl; 
   // Enforce 1 to 1 match
   //Left Keyfeatures
   {
      std::vector<std::vector<std::pair<size_t,size_t> > > temp(pointsRight);
      for (size_t var=0; var<countKeyLeft; ++var){
         for(size_t l=0; l<gm.numberOfLabels(var)-1; ++l)
            temp[labels[var][l]].push_back(std::pair<size_t,size_t>(var,l));
      }
      for(size_t i=0; i<temp.size(); ++i){
         if(temp[i].size()>1){
            for(size_t n=0; n<temp[i].size();++n) 
               for(size_t m=n+1; m<temp[i].size();++m){
                  const size_t var1 = temp[i][n].first;
                  const size_t var2 = temp[i][m].first;
                  const size_t l1 = temp[i][n].second;
                  const size_t l2 = temp[i][m].second;
                  if(var1<var2){
                     if(Adj[var1][var2]!=1){
                        Adj[var1][var2]=1;
                        LabelType nL[2];
                        nL[0] = gm.numberOfLabels(var1);
                        nL[1] = gm.numberOfLabels(var2);
                        pairs[var1][var2] = opengm::ExplicitFunction<ValueType, IndexType, LabelType>(nL,nL+2);
                     }
                     pairs[var1][var2](l1,l2) = 100000.0;
                  }
                  else{
                     if(Adj[var2][var1]!=1){
                        Adj[var2][var1]=1;
                        LabelType nL[2];
                        nL[0] = gm.numberOfLabels(var2);
                        nL[1] = gm.numberOfLabels(var1);
                        pairs[var2][var1] = opengm::ExplicitFunction<ValueType, IndexType, LabelType>(nL,nL+2);
                     }
                     pairs[var2][var1](l2,l1) = 100000.0;
                  }
               }

         }
      }
   }
   //Left Keyfeatures
   {
      std::vector<std::vector<std::pair<size_t,size_t> > > temp(pointsLeft);
      for (size_t var=countKeyLeft; var<countKeyLeft+countKeyRight; ++var){
         for(size_t l=0; l<gm.numberOfLabels(var)-1; ++l)
            temp[labels[var][l]].push_back(std::pair<size_t,size_t>(var,l));
      }
      for(size_t i=0; i<temp.size(); ++i){
         if(temp[i].size()>1){
            for(size_t n=0; n<temp[i].size();++n) 
               for(size_t m=n+1; m<temp[i].size();++m){
                  const size_t var1 = temp[i][n].first;
                  const size_t var2 = temp[i][m].first;
                  const size_t l1 = temp[i][n].second;
                  const size_t l2 = temp[i][m].second;
                  if(var1<var2){
                     if(Adj[var1][var2]!=1){
                        Adj[var1][var2]=1;
                        LabelType nL[2];
                        nL[0] = gm.numberOfLabels(var1);
                        nL[1] = gm.numberOfLabels(var2);
                        pairs[var1][var2] = opengm::ExplicitFunction<ValueType, IndexType, LabelType>(nL,nL+2);
                     }
                     pairs[var1][var2](l1,l2) = 100000.0;
                  }
                  else{
                     if(Adj[var2][var1]!=1){
                        Adj[var2][var1]=1;
                        LabelType nL[2];
                        nL[0] = gm.numberOfLabels(var2);
                        nL[1] = gm.numberOfLabels(var1);
                        pairs[var2][var1] = opengm::ExplicitFunction<ValueType, IndexType, LabelType>(nL,nL+2);
                     }
                     pairs[var2][var1](l2,l1) = 100000.0;
                  }
               }

         }
      }
   }
/*
   count=0;
  for (size_t i=0; i<numVar; ++i){
      for (size_t j=0; j<numVar; ++j){
         std::cout <<Adj[i][j] <<" ";
         count +=Adj[i][j];
      }
      std::cout <<std::endl;
   }
   std::cout<<count <<std::endl; 
*/

   std::cout << "Add pairwise terms ..."<<std::endl;
   for (size_t i=0; i<countKeyLeft+countKeyRight; ++i){

      for (size_t j=0; j<countKeyLeft+countKeyRight; ++j){
         if(Adj[i][j]==1){
            IndexType vars[2];
            vars[0]=i;
            vars[1]=j;
            GmType::FunctionIdentifier currentFunctionIndex = gm.addFunction(pairs[i][j]);
            gm.addFactor(currentFunctionIndex, vars, vars+2);
         }
      }
   }


   // close matching file
   mymatchingfile.close();

   // store gm
   opengm::hdf5::save(gm, opengmfile,"gm");

   // store mapping
   std::ofstream cofile;
   std::string cofilename = opengmfile+".co"; 
   cofile.open(cofilename.c_str());
   for(size_t i=0; i<assignmentsV.size() ;++i){
      cofile << i << "  "<< assignmentsVar[i].first << "  " << assignmentsVar[i].second <<"\n"; 
   }
   cofile.close();

   return 0;
}
