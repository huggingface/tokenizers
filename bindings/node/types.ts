export type TextInputSequence = string
export type PreTokenizedInputSequence = string[]
export type InputSequence = TextInputSequence | PreTokenizedInputSequence

export type TextEncodeInput = TextInputSequence | [TextInputSequence, TextInputSequence]
export type PreTokenizedEncodeInput = PreTokenizedInputSequence | [PreTokenizedInputSequence, PreTokenizedInputSequence]
export type EncodeInput = TextEncodeInput | PreTokenizedEncodeInput
