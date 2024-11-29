#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"

#include "FIInstSelector.h"
#include "FICustomSelectorManager.h"
#include "Utils.h"

#include "CustomTensorOperatorInstSelector.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <cassert>

using namespace llvm;

namespace llfi {

static cl::list< std::string > layerNo_custom("layerNo", cl::desc("Layer Number in \
  which you want to inject bitflip faults. Pass 0 for injecting faults in all the \
  layers.\n Semi-colon seperated values. Example: 1;0;2"), cl::ZeroOrMore);

static cl::list< std::string > layerName_custom("layerName", cl::desc("Layer Name in \
which you want to inject bitflip faults. Semi-colon seperated values. Example: \
Conv;Relu;Pool"), cl::ZeroOrMore);


/**
 * This sample instruction selector only selects instructions in function
 *   main_graph and belonging to the specified tensor operator.
 */
bool CustomTensorOperatorInstSelector::isInstFITarget(Instruction *inst) {
    return checkInCustomTensorOperator(inst, layerNo_custom[0], layerName_custom[0]);
}

bool CustomTensorOperatorInstSelector::shouldInjectInstruction(Instruction *inst) {
    return inst->getOpcode() == Instruction::FAdd ||
            inst->getOpcode() == Instruction::FSub ||
            inst->getOpcode() == Instruction::FMul ||
            inst->getOpcode() == Instruction::FDiv ||
            inst->getOpcode() == Instruction::FCmp;
}

CustomTensorOperatorInstSelector::CustomTensorOperatorInstSelector(){
    inCustomTensorOperator = false;
    injectInAll = false;
    currOperatorName = "";
}

void CustomTensorOperatorInstSelector::getCompileTimeInfo(std::map<std::string, std::string> &info) {
    info["failure_class"] = "HardwareFault";
    info["failure_mode"] = "CustomTensorOperator";
    info["targets"] = "<instructions in main_graph() function and within \
        the specified tensor operator>";
    info["injector"] = "<fi_type>";
}

static RegisterFIInstSelector X("CustomTensorOperator",
                                new CustomTensorOperatorInstSelector());
} // namespace llfi
