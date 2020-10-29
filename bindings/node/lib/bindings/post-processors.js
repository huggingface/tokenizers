const native = require("./native");

module.exports = {
  bertProcessing: native.processors_BertProcessing,
  byteLevelProcessing: native.processors_ByteLevel,
  robertaProcessing: native.processors_RobertaProcessing,
  templateProcessing: native.processors_TemplateProcessing,
};
