% Handle class for openGM potts function
classdef openGMPottsGFunction < openGMFunction
    properties (Hidden, SetAccess = protected, GetAccess = public)
        values;
    end

    properties (Hidden, Constant)
        bellNumbers = [1, 1, 2, 5, 15, 52, 203, 877, 4140];
        maximalOrder = 4; % maximal order currently supported
    end
    
    methods (Access = public)
        function pottsGFunction = openGMPottsGFunction(shape, values)
            % call base constructor
            pottsGFunction = pottsGFunction@openGMFunction(shape);
            % check if data has correct format
            assert(numel(shape) <= openGMPottsGFunction.maximalOrder, strcat('maximal order currently supported is: ', num2str(openGMPottsGFunction.maximalOrder)));
            assert(isvector(shape), 'values has to be a vector.');
            assert(isnumeric(values), 'values have to be numeric');
            assert(numel(values) == openGMPottsGFunction.bellNumbers(numel(shape) + 1), 'number of values has to match the bell number corresponding to the order of the function');
            % copy values
            pottsGFunction.values = values;
        end
    end
    
    methods (Access = protected)
        function clearObsolete(object)
            % Invalidating data as they are no longer required
            object.values = [];
        end
    end
end