import opengm

import sys
from PyQt4 import QtGui

import sys

class Example(QtGui.QWidget):
    
    def __init__(self):
        super(Example, self).__init__()
        #
        self.opengmSolvers=opengm._CppInferenceAlgorithms.inferenceDictNoShort
        self.operator='adder'
        self.accumulator='minimizer'
        self.param_layout =None
        self.vbox=None
        self.updateParam_Button=None
        self.selectedSolverClass=None
        self.selectedSolverParamClass=None
        self.currentParam=None
        self.paramBuffer=dict()
        self.initUI()

        # get initial 
        self.selectedAlgFamily=self.algFamily_ComboBox.currentText()
        self.selectedAlgImpl=self.algImpl_ComboBox.currentText()

        #self.paramLineEditDict=None
    def initUI(self):
        # A ComboBox for all inference algorithms for the given 
        # operator / accumulator combination
        self.algFamily_ComboBox=QtGui.QComboBox(self)
        algForOptAccCombination=self.opengmSolvers[self.operator][self.accumulator]
        for i,algFamily in enumerate(algForOptAccCombination.keys()):
            self.algFamily_ComboBox.insertItem(i,algFamily)
        self.algFamily_ComboBox.activated[str].connect(self.on_alg_changed) 
        
       

        # A ComboBox for all implementations of algorithms
        self.algImpl_ComboBox=QtGui.QComboBox(self)
        algImpls=self.opengmSolvers[self.operator][self.accumulator][str(self.algFamily_ComboBox.currentText())][1]
        for i,algImpl in enumerate(algImpls.keys()):
            self.algImpl_ComboBox.insertItem(i,algImpl)
        self.algImpl_ComboBox.activated[str].connect(self.on_impl_changed) 

        self.on_alg_changed()



        # layoyut
        hbox = QtGui.QHBoxLayout()
        hbox.addStretch(1)
        hbox.addWidget(self.algFamily_ComboBox)
        hbox.addWidget(self.algImpl_ComboBox)
        #box.addWidget(cancelButton)

        self.vbox = QtGui.QVBoxLayout()
        #self.vbox.addStretch(1)
        self.vbox.addLayout(hbox)
        
        #to be delelted
        self.add_parameter_layout(self.vbox,self.selectedSolverParamClass)

        

        #self.add_parameter_layout(self.vbox)

        self.setLayout(self.vbox)    
        
        

        #self.setGeometry(10, 10, 10, 10)
        self.setWindowTitle('Inference Selector')    
        self.show()

    def on_alg_changed(self, text=None):
        self.selectedAlgFamily=str(self.algFamily_ComboBox.currentText())
        # clear current algImpl_ComboBox 
        self.algImpl_ComboBox.clear()
        # update impl. combo box according to the current alg family
        algImpls=self.opengmSolvers[self.operator][self.accumulator][self.selectedAlgFamily][1]
        for i,algImpl in enumerate(algImpls.keys()):
            self.algImpl_ComboBox.insertItem(i,algImpl)
        self.on_impl_changed()
    

    def on_impl_changed(self, text=None):
        self.selectedAlgImpl=str(self.algImpl_ComboBox.currentText())
        solverItem=self.opengmSolvers[self.operator][self.accumulator][self.selectedAlgFamily][1][self.selectedAlgImpl]
        self.selectedSolverClass=solverItem[0]
        self.selectedSolverParamClass=solverItem[1]
        self.clear_layout(self.param_layout)
        self.currentParam=None
        self.add_parameter_layout(self.vbox,self.selectedSolverParamClass)
    

    def _param_name_val_gen(self,solverParamClass,solverParam):
      for propertyName, value in vars(solverParamClass).iteritems(): 
         if( (propertyName.startswith('__') or propertyName.endswith('__')) ==False ):
            #check if it is a property
            if( repr(value).find('property')!=-1):
               attrValue=getattr(solverParam, propertyName)
               yield (propertyName,attrValue)

    def add_parameter_layout(self,parentLayout,parameterClass):
        if(parentLayout is not None):
            self.param_layout = QtGui.QVBoxLayout()
            fullName='Parameter of %s-%s :'%(self.selectedAlgFamily,self.selectedAlgImpl)
            self.param_layout.addWidget(QtGui.QLabel(fullName))
            parentLayout.addStretch(1)
            parentLayout.addLayout(self.param_layout)

            assert (self.currentParam is None)
            self.load_parameter()
            assert (self.currentParam is not None)   
            self.paramLineEditDict=dict()
            for  propertyName, value in self._param_name_val_gen(self.selectedSolverParamClass,self.currentParam):
                subLayout=QtGui.QHBoxLayout()

                (paramAsStr,typeString)=self.value_to_string(value,propertyName)


                label=QtGui.QLabel(propertyName)
                docstring=str(self.selectedSolverParamClass.__dict__[propertyName].__doc__ )
                paramToolTip= '%s\n\nType of the paremeter: %s'% (docstring,typeString)
                if(opengm._is_boost_python_enum(value.__class__)):
                    paramToolTip+=('\n\nAllowed valued for the enum are:\n\n'+self.boost_enum_allowed_values(value.__class__))
                label.setToolTip(paramToolTip)
                lineEdit=QtGui.QLineEdit()
                lineEdit.setText(paramAsStr)
                self.paramLineEditDict[propertyName]=lineEdit


                subLayout.addWidget(label)
                subLayout.addWidget(lineEdit)
                self.param_layout.addLayout(subLayout)

            #update button
            self.updateParam_Button= QtGui.QPushButton('update parameter',self)
            self.resetParam_Button= QtGui.QPushButton('reset parameter',self)
            self.param_layout.addWidget(self.updateParam_Button)
            self.param_layout.addWidget(self.resetParam_Button)
            self.updateParam_Button.clicked.connect(self.update_parameter)
            self.resetParam_Button.clicked.connect(self.reset_parameter)
            #self.updateParam_Button.setFlat(True)
            assert (self.currentParam is not None)
            #self.resize(10,10)

    def boost_enum_value_to_string(self,enumValue):
        return str(enumValue)

    def boost_enum_allowed_values(self,enumClass):
        classDict=enumClass.__dict__
        assert 'names' in classDict
        names=classDict['names']
        retStr=''
        for key in names.keys():
            retStr+=(key+'\n')
        return retStr

    def string_to_boost_python_enum(self,strRep,enumClass):
        classDict=enumClass.__dict__
        assert 'names' in classDict

        lowerRep=strRep.lower()
        lowerRep=lowerRep.replace(' ','')
        lowerRep=lowerRep.replace('\n','')
        names=classDict['names']
        assert isinstance(names,dict)
        #print names
        aKey=None
        for key in names.keys():
            aKey=key
            #print "lowerRep ",lowerRep,' dict name ',str(key)
            if(lowerRep == key.lower()):
                return names[key]
        #todo error
        errorMsg= "\"%s\" is a invalid string representation of an enum,\n the allowed values are:\n\n %s" % (strRep, self.boost_enum_allowed_values(enumClass)   )
        errorMsg+="\nThe value of the enum is will be set to \" %s\""%(str(aKey),)
        msgBox=QtGui.QMessageBox.about(self,"ERROR",errorMsg)
        return names[aKey]

    def value_to_string(self,value,propertyName):
        if(opengm._is_boost_python_enum(value.__class__) ):
            paramAsStr=self.boost_enum_value_to_string(value)
            typeString='enum'
        elif(isinstance(value,opengm.opengmcore._opengmcore.Tribool)):
            typeString='opengm.Tribool'
            strVal=str(value)
            if(strVal=='True'):
                paramAsStr=strVal
            elif(strVal=='False'):
                paramAsStr=strVal
            elif(strVal=='Maybe'):
                paramAsStr='Maybe'
            else:
                assert(false)
        elif(isinstance(value,float)):
            paramAsStr=repr(value)
            typeString='float'
        elif(isinstance(value,int)):
            paramAsStr=repr(value)
            typeString='int'
        elif(isinstance(value,long)):
            paramAsStr=repr(value)
            typeString='long'
        elif(isinstance(value,str)):
            paramAsStr=repr(value)
            typeString='str'
        elif(isinstance(value,bool)):
            paramAsStr=repr(value)
            typeString='bool'
        elif(isinstance(value,tuple)):
            paramAsStr=repr(value)
            typeString='tuple'
        elif(isinstance(value,tuple)):
            paramAsStr=repr(value)
            typeString='tuple'
        elif(propertyName is 'subInfParam'):
            paramAsStr=str(value)
            typeString='subInfParam'
        else:
            raise TypeError('value is of unknown type ',value)
        return (paramAsStr,typeString)

    def load_parameter(self):
        assert(self.selectedSolverParamClass is not None)
        assert(self.selectedAlgFamily is not None)
        assert(self.selectedAlgImpl is not None)
        assert(self.accumulator is not None)
        assert(self.operator is not None)
        key=str(self.selectedAlgFamily)+str(self.selectedAlgImpl)
        if key in self.paramBuffer:
            loadParam=self.paramBuffer[key]
            self.currentParam=loadParam
        else:
            self.currentParam=self.selectedSolverParamClass()
            self.currentParam.set()
    def store_param(self):
        assert(self.selectedSolverParamClass is not None)
        assert(self.selectedAlgFamily is not None)
        assert(self.selectedAlgImpl is not None)
        assert(self.accumulator is not None)
        assert(self.operator is not None)
        assert(self.currentParam is not None)
        key=str(self.selectedAlgFamily)+str(self.selectedAlgImpl)
        self.paramBuffer[key]=self.currentParam
    def reset_parameter(self):
        self.currentParam.set()
        self.update_param_to_text()

    def update_param_to_text(self):
        for  propertyName, value in self._param_name_val_gen(self.selectedSolverParamClass,self.currentParam):
            (paramAsStr,typeString)=self.value_to_string(value,propertyName)
            self.paramLineEditDict[propertyName].setText(paramAsStr)


    def update_text_to_param(self):
        assert (self.currentParam is not None)
        assert (self.selectedSolverParamClass is not None)
        assert (isinstance(self.currentParam,self.selectedSolverParamClass))
        for  propertyName, value in self._param_name_val_gen(self.selectedSolverParamClass,self.currentParam):

            currentText=str(self.paramLineEditDict[propertyName].text())
            try:
                if(opengm._is_boost_python_enum(value.__class__) ):
                    value=self.string_to_boost_python_enum(currentText,value.__class__)
                    setattr(self.currentParam, propertyName, value)
                elif(isinstance(value,opengm.opengmcore._opengmcore.Tribool)):
                    if(currentText=='True'):
                        value=opengm.Tribool(opengm.TriboolStates.true)
                    elif(currentText=='False'):
                        value=opengm.Tribool(opengm.TriboolStates.false)
                    elif(currentText=='Maybe'):
                        value=opengm.Tribool(opengm.TriboolStates.maybe)
                    else :
                        raise ValueError(currentText+' is an invalid value for Tribool,\nallowed values are:\n\nTrue\nFalse\nMaybe')
                    setattr(self.currentParam, propertyName, value)
                elif(isinstance(value,float)):
                    setattr(self.currentParam, propertyName, float(currentText))
                elif(isinstance(value,int)):
                    try:
                        setattr(self.currentParam, propertyName, int(currentText))
                    except:
                        setattr(self.currentParam, propertyName, int(float(currentText)))
                elif(isinstance(value,long)):
                    setattr(self.currentParam, propertyName, long(currentText))
                elif(isinstance(value,str)):
                    setattr(self.currentParam, propertyName, str(currentText))
                elif(isinstance(value,bool)):
                    setattr(self.currentParam, propertyName, bool(currentText))
                elif(isinstance(value,tuple)):
                    setattr(self.currentParam, propertyName, tuple(eval(currentText)))
                elif(isinstance(value,tuple)):
                    setattr(self.currentParam, propertyName, list(eval(currentText)))
                elif(propertyName is 'subInfParam'):
                    kwdict=dict()
                    rawParamStrings=currentText.split(',')
                    for rawString in rawParamStrings:
                        rawString=rawString.replace('\n', '')
                        if(len(rawString)>0):
                            rawStringList=rawString.split('=')
                            assert(len(rawStringList)==2)
                            subPropertyName=rawStringList[0]
                            subPropertyValueStr=rawStringList[1]
                            kwdict[subPropertyName]=eval(subPropertyValueStr)
                    subInfParam=self.currentParam.subInfParam
                    subInfParam.set(**kwdict)
                    self.currentParam.subInfParam=subInfParam
                else:
                    raise TypeError('value is of unknown type ',value)
            except ValueError as detail:
                msgBox=QtGui.QMessageBox.about(self, "ERROR", "Invalid string representation of a parameter:\n %s" % (str(detail),) )
            except TypeError as detail:
                msgBox=QtGui.QMessageBox.about(self, "INTERNAL ERROR", "Invalid type within a parameter object representation of a parameter:\n\n%s" % (str(detail),) )
    def update_parameter(self):
        self.update_text_to_param()
        self.store_param()
        self.update_param_to_text()
    def getInfClassAndParam(self):
        return (self.selectedSolverClass,self.currentParam)

    

    def clear_layout(self, layout):
        if layout is not None:
            while layout.count():
                item = layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clear_layout(item.layout())

def run_selector():
    app = QtGui.QApplication(sys.argv)
    ex = Example()
    app.exec_()
    return ex.getInfClassAndParam()


(infClass,infParam) = run_selector()
print infClass
print infParam
#if __name__ == '__main__':
#    main()
