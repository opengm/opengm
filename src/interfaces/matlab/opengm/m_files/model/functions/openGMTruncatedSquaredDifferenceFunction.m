% Handle class for openGM potts function
classdef openGMTruncatedSquaredDifferenceFunction < openGMFunction
    properties (Hidden, SetAccess = protected, GetAccess = public)
        truncation;
        weight;
    end

    methods (Access = public)
        function tl2Function = openGMTruncatedSquaredDifferenceFunction(shape, truncation, weight)
            % shape has to be two variables
            assert(numel(shape) == 2, 'TruncatedSquaredDifferenceFunction has to be a second order function');
            % call base constructor
            tl2Function = tl2Function@openGMFunction(shape);
            % check if data has correct format
            assert(isnumeric(truncation), 'truncation has to be numeric');
            assert(isnumeric(weight), 'weight has to be numeric');
            % copy values
            tl2Function.truncation = truncation;
            tl2Function.weight = weight;
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