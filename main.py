from run_mdgrow import MDGrow
mdgrow = MDGrow(125,0.674,23,12,)
# mdgrow.dynamics_1(24,prepare=1)
mdgrow.dynamics_mp(1,12*12+1,12)
mdgrow.dynmaics_dp()

mdgrow = MDGrow(250,0.674,23,12)
# mdgrow.dynamics_1(1040,prepare=1)
mdgrow.dynamics_mp(1,12*12+1,12)
mdgrow.dynmaics_dp()

mdgrow = MDGrow(500,0.674,23,12)
# mdgrow.dynamics_1(1,prepare=1)
mdgrow.dynamics_mp(1,12*12+1,12)
mdgrow.dynmaics_dp()

mdgrow = MDGrow(1000,0.674,23,12)
# mdgrow.dynamics_1(1040,prepare=1)
mdgrow.dynamics_mp(1,12*12+1,12)
mdgrow.dynmaics_dp()

mdgrow = MDGrow(2000,0.674,23,12)
# mdgrow.dynamics_1(1040,prepare=1)
mdgrow.dynamics_mp(1,12*12+1,12)
mdgrow.dynmaics_dp()
