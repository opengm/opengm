% Handle class for openGM potts function
classdef openGMPottsFunction < openGMFunction
    properties (Hidden, SetAccess = protected, GetAccess = public)
        valueEqual;     % value if the labels of the two variables are equal
        valueNotEqual;  % value if the labels of the two variables are not equal
    end

    methods (Access = public)
        function pottsFunction = openGMPottsFunction(shape, valueEqual, valueNotEqual)
            % shape has to be two variables
            assert(numel(shape) == 2, 'potts has to be a second order function');
            % call base constructor
            pottsFunction = pottsFunction@openGMFunction(shape);
            % check if data has correct format
            assert(isnumeric(valueEqual), 'valueEqual has to be numeric');
            assert(isnumeric(valueNotEqual), 'valueNotEqual has to be numeric');
            % copy values
            pottsFunction.valueEqual = valueEqual;
            pottsFunction.valueNotEqual = valueNotEqual;
        end
    end
    
    methods (Access = protected)
        function clearObsolete(object)
            % Invalidating data as they are no longer required
            object.valueEqual = [];
            object.valueNotEqual = [];
        end
    end
end