% Handle class for openGM function
classdef openGMFunction < handle
    properties (Hidden, SetAccess = protected, GetAccess = public)
        % stores function ID if the function after it has been asigned to
        % a model.
        functionID;
        % stores shape information of the function.
        shape;
        % stores information if function has already been assigned to a
        % model.
        assigned;
    end
    
    methods (Access = public)
        % Constructor
        function obj = openGMFunction(shape)
            % check if shape has correct format
            assert(isvector(shape), 'shape has to be a vector.');
            assert(isnumeric(shape), 'shape has to be numeric.');
            
            obj.shape = shape;
            obj.assigned = false;
        end
        
        % Get function ID if function has already been assigned to a model.
        function functionID = getFunctionID(object)
            % check if function has already be assigned to a model
            assert(object.assigned, 'Function has not yet been assigned to a model.');
            functionID = object.functionID;
        end
        
        % Get shape of the function
        function shape = getShape(object)
            shape = object.shape;
        end
        
        % Set function ID for the function. Will be called by openGMModel
        % when the function is added to the model.
        function setFunctionID(object, functionID)
            % check if function has not yet been assigned to a model
            assert(~object.assigned, 'Function has already been assigned to a model.');
            % check if functionID has correct format
            assert(isinteger(functionID), 'Wrong format for functionID.');
            assert(isequal(size(functionID),[1, 2]), 'Wrong format for functionID.');
            
            object.functionID = functionID;
            object.assigned = true;
            object.clearObsolete();
        end
    end
    
    methods (Abstract, Access = protected)
        % Will be called after function is added to a model. Thus most
        % information stored on matlab side (e.g. value tables) are
        % redundant and are no longer required. This helps to keep required
        % storage at a minimum
        clearObsolete(object);
    end
end