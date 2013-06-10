function modelinfo2xml( modelDir, resultLocation )
%modelinfo2xml Retrieve informations about opengm models and store them in
%a single xml file
%   Detailed explanation goes here
   
    % save current working directory
    oldDir = pwd;
    % change working direktory
    cd(modelDir);
    % get all model files
    models = dir('*.h5');
    
    % get current folder name
    [~, currentFolder, ~] = fileparts(modelDir);
    
    % set values
    tag = currentFolder;
    thumb = '?';
    modelname = currentFolder;
    author = '?';
    converter = '?';
    minvariables = Inf;
    maxvariables = -Inf;
    minlabels = Inf;
    maxlabels = -Inf;
    minorder = Inf;
    maxorder = -Inf;
    structure = '?';
    functions = '';
    instances = numel(models);
    reference = '?';
    zip = '?';
    comment = '?';
    
    % get values from model
    for i = 1:instances
        % load model
        currentModel = openGMModel('load', models(i).name);
        
        % check num variables
        numVariables = currentModel.numberOfVariables;
        if(numVariables < minvariables)
            minvariables = numVariables;
        end
        if(numVariables > maxvariables)
            maxvariables = numVariables;
        end
        
        % check num labels
        maxNumLabels = currentModel.maximumLabelOrder;
        if(maxNumLabels < minlabels)
            minlabels = maxNumLabels;
        end
        if(maxNumLabels > maxlabels)
            maxlabels = maxNumLabels;
        end
        
        % check order
        maxFactorOrder = currentModel.maximumFactorOrder;
        if(maxFactorOrder < minorder)
            minorder = maxFactorOrder;
        end
        if(maxFactorOrder > maxorder)
            maxorder = maxFactorOrder;
        end
        
        % check structure
        if(currentModel.hasGridStructure)
            structure = 'grid';
        end
        
        % check factor types
        if(currentModel.hasPotts)
            if(isempty(functions))
                functions = 'potts';
            else
                if(isempty(strfind(functions, 'potts')))
                    functions = [functions ', potts'];
                end
            end
        end
        if(currentModel.hasTruncatedAbsoluteDifference)
            if(isempty(functions))
                functions = 'TL1';
            else
                if(isempty(strfind(functions, 'TL1')))
                    functions = [functions ', TL1'];
                end
            end
        end
        if(currentModel.hasTruncatedSquaredDifference)
            if(isempty(functions))
                functions = 'TL2';
            else
                if(isempty(strfind(functions, 'TL2')))
                    functions = [functions ', TL2'];
                end
            end
        end
    end

    % check if functions were set
    if(isempty(functions))
        functions = '?';
    end
    
    % create xml DOM
    modelNode = com.mathworks.xml.XMLUtils.createDocument('model');
    model = modelNode.getDocumentElement;

    % tag
    addElement('tag', tag);
    % thumb
    addElement('thumb', thumb);
    % modelname
    addElement('modelname', modelname);
    % author
    addElement('author', author);
    % converter
    addElement('converter', converter);
    % minvariables
    addElement('minvariables', minvariables);
    % maxvariables
    addElement('maxvariables', maxvariables);
    % minlabels
    addElement('minlabels', minlabels);
    % maxlabels
    addElement('maxlabels', maxlabels);
    % minorder
    addElement('minorder', minorder);
    % maxorder
    addElement('maxorder', maxorder);
    % structure
    addElement('structure', structure);
    % functions
    addElement('functions', functions);
    % instances
    addElement('instances', instances);
    % reference
    addElement('reference', reference);
    % zip
    addElement('zip', zip);
    % comment
    addElement('comment', comment);

    % export xml DOM
    xmlwrite(resultLocation,modelNode);

    % return to old working directory
    cd(oldDir);

    function addElement( elementName, elementValue )
        if(isnumeric(elementValue)) 
            elementValue = num2str(elementValue);
        end
        % local function, has access to all elements of function modelinfo2xml
        newNode = modelNode.createElement(elementName);
        newNode.appendChild(modelNode.createTextNode(elementValue));
        model.appendChild(newNode);
    end
end

