def DLprediction2F(Xin, yin, DLmodel, modelflag, delta_input):
    # Input is the windows [Num_Seq] [Nloc] [Tseq] [NpropperseqTOT] (SymbolicWindows False)
    # Input is  the sequences [Nloc] [Num_Time-1] [NpropperseqTOT] (SymbolicWindows True)
    # Input Predictions are always [Num_Seq] [NLoc] [NpredperseqTOT]
    # Label Array is always [Num_Seq][Nloc] [0=Window(first sequence)#, 1=Location]

    if GarbageCollect:
        gc.collect()
    global OuterBatchDimension, Nloc_sample, d_sample, max_d_sample

    SensitivityAnalyze = np.full((NpropperseqTOT), False, dtype=np.bool)
    SensitivityChange = np.zeros((NpropperseqTOT), dtype=np.float32)
    SensitvitybyPrediction = False
    if ReadDecember2021:
        for iprop in range(0, NpropperseqTOT):
            SensitivityAnalyze[iprop] = True

    something = 0  # Why name it something??
    SensitivityList = []  # A list of integers from 0 to 21?
    for iprop in range(0, NpropperseqTOT):
        if SensitivityAnalyze[iprop]:
            something += 1
            SensitivityList.append(iprop)
    if something == 0:
        return

    # Remove the parameters / cases and deaths
    SensitivityList = SensitivityList[2:-3]
    something -= 5

    SampleSize = 1
    Delta = delta_input
    DeltaTotal = []

    SensitivityFitPredictions = np.zeros([Num_Seq, Nloc, NpredperseqTOT, 1 + something], dtype=np.float32)
    FRanges = np.full((NpredperseqTOT), 1.0, dtype=np.float32)
    current_time = timenow()
    print(wraptotext(startbold + startred + 'DLPrediction2F ' + current_time + ' ' + RunName + resetfonts))

    sw = np.empty_like(yin, dtype=np.float32)  # What is this?
    for i in range(0, sw.shape[0]):
        for j in range(0, sw.shape[1]):
            for k in range(0, NpredperseqTOT):
                sw[i, j, k] = Predictionwgt[k]
    labelarray = np.empty([Num_Seq, Nloc, 2], dtype=np.int32)
    for iseq in range(0, Num_Seq):
        for iloc in range(0, Nloc):
            labelarray[iseq, iloc, 0] = iseq
            labelarray[iseq, iloc, 1] = iloc

    Totaltodo = Num_Seq * Nloc
    Nloc_sample = Nloc  # default

    if IncreaseNloc_sample > 1:  # What does these two do?
        Nloc_sample = int(Nloc_sample * IncreaseNloc_sample)
    elif DecreaseNloc_sample > 1:
        Nloc_sample = int(Nloc_sample / DecreaseNloc_sample)

    if Totaltodo % Nloc_sample != 0:
        printexit('Invalid Nloc_sample ' + str(Nloc_sample) + " " + str(Totaltodo))
    d_sample = Tseq * Nloc_sample  # = 13 * num_locations?
    max_d_sample = d_sample
    OuterBatchDimension = int(Totaltodo / Nloc_sample)
    print(' Predict with ' + str(Nloc_sample) + ' sequences per sample and batch size ' + str(OuterBatchDimension))

    print(startbold + startred + 'Sensitivity using Delta ' + str(Delta) + resetfonts)
    for Sensitivities in range(0, 1 + something):
        if Sensitivities == 0:  # Here is the baseline
            iprop = -1
            print(startbold + startred + 'Basic Predictions' + resetfonts)
            if SymbolicWindows:
                ReshapedSequencesTOTmodified = ReshapedSequencesTOT  # (NOT used if modelflag == 2) Why ??
                if modelflag == 2:
                    DLmodel.MakeMapping()
            else:
                Xinmodified = Xin
        else:  # For every other properties
            iprop = SensitivityList[Sensitivities - 1]
            maxminplace = PropertyNameIndex[iprop]
            lastline = ''
            if iprop < Npropperseq:
                lastline = ' Normed Mean ' + str(round(QuantityStatistics[maxminplace, 5], 4))
            print(startbold + startred + 'Property ' + str(iprop) + ' ' + InputPropertyNames[
                maxminplace] + resetfonts + lastline)
            if SymbolicWindows:
                if modelflag == 2:
                    DLmodel.SetupProperty(iprop)
                    delta_sum = DLmodel.ChangeProperty(Delta)
                    DLmodel.MakeMapping()
                    DeltaTotal.append(delta_sum)
                else:
                    ReshapedSequencesTOTmodified = np.copy(ReshapedSequencesTOT)
                    # print(ReshapedSequencesTOTmodified.shape) [65, 503, 22]
                    ReshapedSequencesTOTmodified[:, :, iprop] = Delta + ReshapedSequencesTOTmodified[:, :, iprop]
            else:
                Xinmodified = np.copy(Xin)
                Xinmodified[:, :, :, iprop] = Delta + Xinmodified[:, :, :, iprop]
        CountFitPredictions = np.zeros([Num_Seq, Nloc, NpredperseqTOT], dtype=np.float32)
        meanvalue2 = 0.0
        meanvalue3 = 0.0
        meanvalue4 = 0.0
        variance2 = 0.0
        variance3 = 0.0
        variance4 = 0.0

        for shuffling in range(0, SampleSize):
            if GarbageCollect:
                gc.collect()
            yuse = yin
            labeluse = labelarray
            y2 = np.reshape(yuse, (-1, NpredperseqTOT)).copy()
            labelarray2 = np.reshape(labeluse, (-1, 2))

            if SymbolicWindows:
                # Xin X2 X3 not used rather ReshapedSequencesTOT
                labelarray2, y2 = shuffleDLinput(labelarray2, y2)
                ReshapedSequencesTOTuse = ReshapedSequencesTOTmodified
            else:
                Xuse = Xinmodified
                X2 = np.reshape(Xuse, (-1, Tseq, NpropperseqTOT)).copy()
                X2, y2, labelarray2 = shuffleDLinput(X2, y2, labelarray2)
                X3 = np.reshape(X2, (-1, d_sample, NpropperseqTOT))

            y3 = np.reshape(y2, (-1, Nloc_sample, NpredperseqTOT))
            sw = np.reshape(sw, (-1, Nloc_sample, NpredperseqTOT))
            labelarray3 = np.reshape(labelarray2, (-1, Nloc_sample, 2))

            quan2 = 0.0
            quan3 = 0.0
            quan4 = 0.0
            for Batchindex in range(0, OuterBatchDimension):
                if GarbageCollect:
                    gc.collect()

                if SymbolicWindows:  # modelflag = 2
                    if modelflag == 2:  # Note first index of InputVector Location, Second is sequence number; labelarray3 is opposite
                        InputVector = np.empty((Nloc_sample, 2), dtype=np.int32)
                        for iloc_sample in range(0, Nloc_sample):
                            InputVector[iloc_sample, 0] = labelarray3[Batchindex, iloc_sample, 1]
                            InputVector[iloc_sample, 1] = labelarray3[Batchindex, iloc_sample, 0]
                    else:
                        X3local = list()
                        for iloc_sample in range(0, Nloc_sample):
                            LocLocal = labelarray3[Batchindex, iloc_sample, 1]
                            SeqLocal = labelarray3[Batchindex, iloc_sample, 0]
                            X3local.append(ReshapedSequencesTOTuse[LocLocal, SeqLocal:SeqLocal + Tseq])
                        InputVector = np.array(X3local)
                else:
                    InputVector = X3[Batchindex]

                Labelsused = labelarray3[Batchindex]
                Time = None
                if modelflag == 0:
                    InputVector = np.reshape(InputVector, (-1, Tseq, NpropperseqTOT))
                elif modelflag == 1:
                    Time = SetSpacetime(np.reshape(Labelsused[:, 0], (1, -1)))
                    InputVector = np.reshape(InputVector, (1, Tseq * Nloc_sample, NpropperseqTOT))

                PredictedVector = DLmodel(InputVector, training=PredictionTraining, Time=Time)
                PredictedVector = np.reshape(PredictedVector, (1, Nloc_sample, NpredperseqTOT))

                swbatched = sw[Batchindex, :, :]
                if LocationBasedValidation:
                    swT = np.zeros([1, Nloc_sample, NpredperseqTOT], dtype=np.float32)
                    swV = np.zeros([1, Nloc_sample, NpredperseqTOT], dtype=np.float32)
                    for iloc_sample in range(0, Nloc_sample):
                        fudgeT = Nloc / TrainingNloc
                        fudgeV = Nloc / ValidationNloc
                        iloc = Labelsused[iloc_sample, 1]
                        if MappingtoTraining[iloc] >= 0:
                            swT[0, iloc_sample, :] = swbatched[iloc_sample, :] * fudgeT
                        else:
                            swV[0, iloc_sample, :] = swbatched[iloc_sample, :] * fudgeV
                TrueVector = y3[Batchindex]
                TrueVector = np.reshape(TrueVector, (1, Nloc_sample, NpredperseqTOT))
                swbatched = np.reshape(swbatched, (1, Nloc_sample, NpredperseqTOT))

                losspercall = numpycustom_lossGCF1(TrueVector, PredictedVector, swbatched)
                quan2 += losspercall
                # bbar.update(1)
                if LocationBasedValidation:
                    losspercallTr = numpycustom_lossGCF1(TrueVector, PredictedVector, swT)
                    quan3 += losspercallTr
                    losspercallVl = numpycustom_lossGCF1(TrueVector, PredictedVector, swV)
                    quan4 += losspercallVl

                for iloc_sample in range(0, Nloc_sample):
                    LocLocal = Labelsused[iloc_sample, 1]
                    SeqLocal = Labelsused[iloc_sample, 0]
                    yyhat = PredictedVector[0, iloc_sample]
                    CountFitPredictions[SeqLocal, LocLocal, :] += FRanges  # What are these two?
                    SensitivityFitPredictions[SeqLocal, LocLocal, :, Sensitivities] += yyhat

                fudge = 1.0 / (1.0 + Batchindex)
                mean2 = quan2 * fudge
                if LocationBasedValidation:
                    mean3 = quan3 * fudge
                    mean4 = quan4 * fudge

                    # Processing at the end of Sampling Loop
            fudge = 1.0 / OuterBatchDimension
            quan2 *= fudge
            quan3 *= fudge
            quan4 *= fudge
            meanvalue2 += quan2
            variance2 += quan2 ** 2
            variance3 += quan3 ** 2
            variance4 += quan4 ** 2
            if LocationBasedValidation:
                meanvalue3 += quan3
                meanvalue4 += quan4

        if Sensitivities == 0:
            iprop = -1
            lineend = startbold + startred + 'Basic Predictions' + resetfonts
        else:
            iprop = SensitivityList[Sensitivities - 1]
            nameplace = PropertyNameIndex[iprop]
            maxminplace = PropertyAverageValuesPointer[iprop]
            lastline = ' Normed Mean ' + str(round(QuantityStatistics[maxminplace, 5], 4))
            lineend = startbold + startred + 'Property ' + str(iprop) + ' ' + InputPropertyNames[
                nameplace] + resetfonts + lastline
            if modelflag == 2:
                DLmodel.ResetProperty()

        meanvalue2 /= SampleSize

        global GlobalTrainingLoss, GlobalValidationLoss, GlobalLoss
        meanvalue2 /= SampleSize
        GlobalLoss = meanvalue2
        GlobalTrainingLoss = 0.0
        GlobalValidationLoss = 0.0

        if LocationBasedValidation:
            meanvalue3 /= SampleSize
            meanvalue4 /= SampleSize
            GlobalTrainingLoss = meanvalue3
            GlobalValidationLoss = meanvalue4

        # Sequence Location Predictions
        SensitivityFitPredictions[:, :, :, Sensitivities] = np.divide(SensitivityFitPredictions[:, :, :, Sensitivities],
                                                                      CountFitPredictions[:, :, :])
        
        if Sensitivities == 0:
            Goldstandard = np.sum(np.abs(SensitivityFitPredictions[:, :, :, Sensitivities]), axis=(0, 1))
            TotalGS = np.sum(Goldstandard)
            continue

        Change = np.sum(np.abs(
            np.subtract(SensitivityFitPredictions[:, :, :, Sensitivities], SensitivityFitPredictions[:, :, :, 0])),
                        axis=(0, 1))
        TotalChange = np.sum(Change)
        SensitivityChange[iprop] = TotalChange
        print(str(round(TotalChange, 5)) + ' GS ' + str(round(TotalGS, 5)) + ' ' + lineend)
        if SensitvitybyPrediction:
            for ipred in range(0, NpredperseqTOT):
                print(str(round(Change[ipred], 5)) + ' GS ' + str(round(Goldstandard[ipred], 5))
                      + ' ' + str(ipred) + ' ' + Predictionname[ipred] + ' wgt ' + str(round(Predictionwgt[ipred], 3)))

    # ---------------- End of the function -------------#
    print(startbold + startred + '\nSummarize Changes Total ' + str(round(TotalGS, 5)) + ' Delta ' + str(
        Delta) + resetfonts)
    for Sensitivities in range(1, 1 + something):
        iprop = SensitivityList[Sensitivities - 1]
        nameplace = PropertyNameIndex[iprop]
        maxminplace = PropertyAverageValuesPointer[iprop]

        lastline = ' Normed Mean ' + str(round(QuantityStatistics[maxminplace, 5], 4))
        lastline += ' Normed Std ' + str(round(QuantityStatistics[maxminplace, 6], 4))

        TotalChange = SensitivityChange[iprop]

        # Total number of Deltas applied = Num_nonzero * Delta
        NormedChange = TotalChange / (DeltaTotal[Sensitivities - 1])

        stdmeanratio = 0.0
        stdchangeratio = 0.0
        if np.abs(QuantityStatistics[maxminplace, 5]) > 0.0001:
            stdmeanratio = QuantityStatistics[maxminplace, 6] / QuantityStatistics[maxminplace, 5]
        if np.abs(QuantityStatistics[maxminplace, 6]) > 0.0001:
            stdchangeratio = NormedChange / QuantityStatistics[maxminplace, 6]

        lratios = ' Normed Change ' + str(round(NormedChange, 5)) + ' /std ' + str(round(stdchangeratio, 5))
        lratios += ' Delta Total ' + str(DeltaTotal[Sensitivities - 1])
        lratios += ' Std/Mean ' + str(round(stdmeanratio, 5))
        print(str(iprop) + ' Change ' + str(round(TotalChange, 2)) + startbold + lratios
              + ' ' + InputPropertyNames[nameplace] + resetfonts + lastline)

    current_time = timenow()
    print(startbold + startred + '\nEND DLPrediction2F ' + current_time + ' ' + RunName + resetfonts)
    return