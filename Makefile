
C1P1_ID = 1LFe2NhXLJ0FzStLvgjrh8aWE-PbN7BIz
C1P1_FN = c1p1/C1-P1_Train_Dev_fixed.rar

C1P2_ID = 1hPRbUqOBvIC9HHqcA42r1F3PlIxmPOF8
C1P2_FN = c1p2/C1-P2_Train_Dev.rar

c1p1: $(C1P1_FN)
	bash gdown.sh $(C1P1_ID) $(C1P1_FN)
	rar x $(C1P1_FN) `dirname $(C1P1_FN)`

c1p2: $(C1P2_FN)
	bash gdown.sh $(C1P2_ID) $(C1P2_FN)
	rar x $(C1P2_FN) `dirname $(C1P2_FN)`
