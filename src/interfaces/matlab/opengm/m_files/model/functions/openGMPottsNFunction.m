% Handle class for openGM potts function
classdef openGMPottsNFunction < openGMFunction
    properties (Hidden, SetAccess = protected, GetAccess = public)
        valueEqual;     % value if the labels of the two variables are equal
        valueNotEqual;  % value if the labels of the two variables are not equal
    end

    methods (Access = public)
        function pottsNFunction = openGMPottsNFunction(shape, valueEqual, valueNotEqual)
            % call base constructor
            pottsNFunction = pottsNFunction@openGMFunction(shape);
            % check if data has correct format
            assert(isnumeric(valueEqual), 'valueEqual has to be numeric');
            assert(isnumeric(valueNotEqual), 'valueNotEqual has to be numeric');
            % copy values
            pottsNFunction.valueEqual = valueEqual;
            pottsNFunction.valueNotEqual = valueNotEqual;
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