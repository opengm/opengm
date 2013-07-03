% Handle class for openGM potts function
classdef openGMTruncatedAbsoluteDifferenceFunction < openGMFunction
    properties (Hidden, SetAccess = protected, GetAccess = public)
        truncation;
        weight;
    end

    methods (Access = public)
        function tl1Function = openGMTruncatedAbsoluteDifferenceFunction(shape, truncation, weight)
            % shape has to be two variables
            assert(numel(shape) == 2, 'TruncatedAbsoluteDifferenceFunction has to be a second order function');
            % call base constructor
            tl1Function = tl1Function@openGMFunction(shape);
            % check if data has correct format
            assert(isnumeric(truncation), 'truncation has to be numeric');
            assert(isnumeric(weight), 'weight has to be numeric');
            % copy values
            tl1Function.truncation = truncation;
            tl1Function.weight = weight;
        end
    end
    
    methods (Access = protected)
        function clearObsolete(object)
            % Invalidating data as they are no longer required
            object.truncation = [];
            object.weight = [];
        end
    end
end