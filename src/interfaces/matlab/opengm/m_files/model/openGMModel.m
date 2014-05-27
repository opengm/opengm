% Handle class for openGM model
classdef openGMModel < handle
    
    properties (Constant, Hidden)
        supportedFunctionTypes = {'openGMExplicitFunction', 'openGMPottsFunction', 'openGMPottsNFunction', 'openGMPottsGFunction', 'openGMTruncatedSquaredDifferenceFunction', 'openGMTruncatedAbsoluteDifferenceFunction'};
    end
    
    properties (Constant, Hidden)
        % necessary mex file list for destructor to avoid double deletion
        % of model.        
        expectedMexFunctions = {'newModel', 'deleteModel', 'loadModel', 'storeModel', 'addVariables', 'addFactor', 'addFactors', 'addUnaries', 'addFunction', 'numFactors', 'numLabels', 'numVariables', 'opengm'}
    end
    
    properties (Hidden, SetAccess = protected, GetAccess = public)
        % stores handle for C++ object
        modelHandle;
        % value type for openGM model
        ValueType;
        % index type for openGM model
        IndexType;
        % label type for openGM model
        LabelType;
        % space type for openGM model
        SpaceType;
        % operator type for openGM model
        OperatorType;
        % function type list for openGM model
        FunctionTypeList;
    end

    properties (SetAccess = protected, GetAccess = protected)

    end
    methods (Access = public)
        % constructor
        function model = openGMModel(varargin)
            p = inputParser;

            defaultLoad = '';
            addParamValue(p,'load',defaultLoad,...
                          @(x) ischar(x));
            defaultDataset = 'gm';
            addParamValue(p,'dataset',defaultDataset,...
                          @(x) ischar(x));
                      
            parse(p,varargin{:});      
            
            if(~ismember('load', p.UsingDefaults))
                % load model on construction
                model.modelHandle = 0;
                model.load(p.Results.load, p.Results.dataset);
            else
                model.modelHandle = newModel();
            end
        end
        
        % destructor
        function delete(model)
            % detect if any mex file for opengm is loaded 
            % if non is loaded, the model has alredy been deleted
            [~, x] = inmem;            
            if(any(any(ismember(x, model.expectedMexFunctions))))
                if(model.modelHandle ~= 0) 
                    deleteModel(model.modelHandle);
                end
            end
        end
        
        % store model
        function store(model, fileLocation, dataset)
            storeModel(fileLocation, dataset, model.modelHandle);
        end
        
        % load model
        function load(model, fileLocation, dataset)
            if(model.modelHandle ~= 0)
                deleteModel(model.modelHandle);
            end
            model.modelHandle = loadModel(fileLocation, dataset);
        end
     
        % add variables
        function addVariables(model, numbersOfLabels)
            addVariables(model.modelHandle, numbersOfLabels);
        end
        
        % add function
        function addFunction(model, functionIn)
            assert(any(strcmp(class(functionIn), model.supportedFunctionTypes)), 'No supported function Type');
            functionIn.setFunctionID(addFunction(model.modelHandle, functionIn));
        end
        
        % add factor
        function addFactor(model, functionIn, variables)
            assert(any(strcmp(class(functionIn), model.supportedFunctionTypes)), 'No supported function Type');
            addFactor(model.modelHandle, functionIn.getFunctionID(), variables);
        end
        
        % add multiple factors
        function addFactors(model, functionIn, variables)
            assert(any(strcmp(class(functionIn), model.supportedFunctionTypes)), 'No supported function Type');
            addFactors(model.modelHandle, functionIn.getFunctionID(), variables);
        end
        
        % add multiple unaries
        function addUnaries(model, variableIDs, functionValues)
            addUnaries(model.modelHandle, variableIDs, functionValues);
        end  

        % add multiple pairwise terms
        function addPairwiseTerms(model, variableIDs, functionValues)
            addPairwiseTerms(model.modelHandle, variableIDs, functionValues);
        end
                        
        % number of variables
        function numOfVariables = numberOfVariables(model)
            numOfVariables = numVariables(model.modelHandle);
        end
        
        % number of factors
        function numOfFactors = numberOfFactors(model)
            numOfFactors = numFactors(model.modelHandle);
        end
        
        % max factor order
        function maxFactorOrder = maximumFactorOrder(model)
            maxFactorOrder = factorOrder(model.modelHandle);
        end
        
        % max label order
        function labelOrder = maximumLabelOrder(model)
            labelOrder = maxLabelOrder(model.modelHandle);
        end
        
        % number of labels for a given variable
        function numOfLabels = numberOfLabels(model, varIndex)
            assert(numel(varIndex) == 1, 'variable index has to be a scalar');
            assert(isnumeric(varIndex), 'variable index has to be numeric');
            numOfLabels = numLabels(model.modelHandle, varIndex);
        end
        
        % value table for a given factor
        function [factorTable, variables] = getFactorTable(model, factorIndex)
            assert(numel(factorIndex) == 1, 'factor index has to be a scalar');
            assert(isnumeric(factorIndex), 'factor index has to be numeric');
            assert(factorIndex < model.numberOfFactors(), 'factor index has to be smaler than the total number of factors');
            [factorTable, variables] = getFactorTable(model.modelHandle, factorIndex);
        end
        
        % is grid
        function isgrid = hasGridStructure(model)
            isgrid = isGrid(model.modelHandle);
        end
        
        % has at least one potts factor
        function haspotts = hasPotts(model)
            haspotts = hasPottsFactor(model.modelHandle);
        end
        
        % has at least one TL1 factor
        function hastl1 = hasTruncatedAbsoluteDifference(model)
            hastl1 = hasTruncatedAbsoluteDifferenceFactor(model.modelHandle);
        end
        
        % has at least one TL2 factor
        function hastl2 = hasTruncatedSquaredDifference(model)
            hastl2 = hasTruncatedSquaredDifferenceFactor(model.modelHandle);
        end
    end
end
