analysis_HR = Model.AddHarmonicResponseAnalysis()

analysis_HR

analysis_settings_HR = analysis_HR.AnalysisSettings
analysis_HR.PropertyByName("HarmonicForcingFrequencyMax").InternalValue = 10
analysis_HR.PropertyByName("HarmonicForcingFrequencyIntervals").InternalValue = 1
analysis_settings_HR.PropertyByName("HarmonicSolutionMethod").InternalValue = 1



remote_force = analysis_HR.AddRemoteForce()
remote_force.DefineBy = LoadDefineBy.Components
remote_force.XComponent.Output.SetDiscreteValue(0, Quantity(5, "N"))
remote_force.YComponent.Output.SetDiscreteValue(0, Quantity(10, "N"))
remote_force.ZComponent.Output.SetDiscreteValue(0, Quantity(15, "N"))
remote_force.XPhaseAngle.Output.SetDiscreteValue(0, Quantity(30, "deg"))
remote_force.YPhaseAngle.Output.SetDiscreteValue(0, Quantity(60, "deg"))
remote_force.ZPhaseAngle.Output.SetDiscreteValue(0, Quantity(90, "deg"))

'''NOTE : All workflows will not be recorded, as recording is under development.'''


#region Details View Action
remote_force_30 = DataModel.GetObjectById(30)
remote_force_30.XComponent.Output.SetDiscreteValue(0, Quantity(5, "N"))
#endregion

#region Unpublished API
import context_menu1
context_menu.DoChangeEDObjectSelection(ExtAPI,'_EnggData_Table')
#endregion

#region Details View Action
remote_force_30.XPhaseAngle.Output.SetDiscreteValue(0, Quantity(30, "deg"))
#endregion
