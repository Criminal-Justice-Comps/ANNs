readme.txt

Problems:
  In Data:
    Some features have to do with recidivism, so they cannot be included
    c_charge_desc causes problems--not sure if we should include it in its current form
        Temporarily excluded from Testing
    validation contains some values not in training--how to solve this problem?
    always predict recidivism--how to solve this problem?
    POTENTIAL PROBLEM:
      In data -- lopsided class sizes--how to deal with this?


Findings:
  With
    batch_size = 128
    epochs = 25
  Here are our initial findings on the toy data:
    Perceptron:
      Loss is 0.552

    With hidden layers:
      # hidden layers | loss
      ----------------|----------
            1         | 0.5522
            2         | 0.5522
            3         | 0.5522
            10        | 0.5522
