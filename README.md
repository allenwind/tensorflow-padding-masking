# tensorflow-padding-masking

NLP中面对的数据是不定长的序列，padding和masking变得尤为重要，这里提供padding和masking的例子以及常用层的mask支持。

常用层的mask支持包括：

- MaskGlobalAveragePooling1D
- MaskGlobalMaxPooling1D
- MaskGlobalMaxMinPooling1D
- AttentionPooling1D
- MaskBiLSTM
- MaskConv1D
