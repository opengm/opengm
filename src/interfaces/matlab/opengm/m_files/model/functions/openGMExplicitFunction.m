% Handle class for openGM explicit function
classdef openGMExplicitFunction < openGMFunction
    properties (Hidden, SetAccess = protected, GetAccess = public)
        functionValues;
    end
    methods (Access = public)
        function explicitFunction = openGMExplicitFunction(shape, functionValues)
            % call base constructor
            explicitFunction = explicitFunction@openGMFunction(shape);
            % check if data has correct format
            assert(isnumeric(functionValues), 'functionValues has to be numeric');
            assert(prod(shape) == numel(functionValues), 'number of elements of functionValues has to be suitable for the given shape');
            % copy values
            explicitFunction.functionValues = functionValues;
        end
    end
    
    methods (Access = protected)
        function clearObsolete(object)
            % Invalidating data as they are no longer required
            object.functionValues = [];
        end
    end
end