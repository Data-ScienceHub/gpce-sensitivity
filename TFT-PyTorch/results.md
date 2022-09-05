# Primary Split Results
The training was run for maximum 60 epochs. Outliers removed from both inputs and targets. Use the configuration file [total_early_stopped_target_cleaned_scaled](configurations/total_early_stopped_target_cleaned_scaled.json) to reproduce the results.

##  Train

![daily-cases](results/TFT/figures_best/Summed_plot_Cases_Train.jpg)
![daily-deaths](results/TFT/figures_best/Summed_plot_Deaths_Train.jpg)

##  Validation

![daily-cases](results/TFT/figures_best/Summed_plot_Cases_Validation.jpg)
![daily-deaths](results/TFT/figures_best/Summed_plot_Deaths_Validation.jpg)

##  Test

![daily-cases](results/TFT/figures_best/Summed_plot_Cases_Test.jpg)
![daily-deaths](results/TFT/figures_best/Summed_plot_Deaths_Test.jpg)

##  Attention on prior days (train data)

The closer the past day is to the present day, the more attention weight it has. Also the same weekday in the previous week (Time index -7), has similary high weight as the previous day (Time index -1). So yesterday's data and same weekday's data from previous week are most important for model prediction.

![train-attention](results/TFT/figures_best/Train_attention.jpg)

##  Weekly attention

Attention weights mostly peak on Wednesday, as seen from the mean values. And it is lowest on Saturday/Sunday. This is because covid cases often peaked at Thursday or Friday. On weekends less cases and deaths are reported, so eventually they have less impact on the model attention.

Train ![train-weekly-attention](results/TFT/figures_best/Train_weekly_attention.jpg)

##  Variable importance (Train data)

Weeekyl patterns are very important as seen from the attentions. That is also evident from the variable importance, as weekly features get most importance. Then past observations of the target variables. Since covid cases/deaths in the past weeks can help learn the trend and predict the future better.

* Static variables ![Train_static_variables](results/TFT/figures_best/Train_static_variables.jpg)
* Encoder variables ![Train_encoder_variables](results/TFT/figures_best/Train_encoder_variables.jpg)
* Decoder variables ![Train_decoder_variables](results/TFT/figures_best/Train_decoder_variables.jpg)