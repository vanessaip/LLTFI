#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"

#include "FIInstSelector.h"
#include "FICustomSelectorManager.h"

#include "CustomTensorOperatorInstSelector.h"

#include <vector>
using namespace llvm;

namespace llfi {

static cl::list< std::string > layerNo_WandB("layerNo_WandB", cl::desc("Layer Number in \
  which you want to inject bitflip faults. Pass 0 for injecting faults in all the \
  layers.\n Semi-colon seperated values. Example: 1;0;2"), cl::ZeroOrMore);

static cl::list< std::string > layerName_WandB("layerName_WandB", cl::desc("Layer Name in \
which you want to inject bitflip faults. Semi-colon seperated values. Example: \
Conv;Relu;Pool"), cl::ZeroOrMore);

static cl::opt< bool > bias("bias", cl::desc("Include injecting into biases."), cl::init(false));

static cl::opt< bool > weights("weights", cl::desc("Include injecting into weights."), cl::init(true));

class WeightsAndBiasesInstSelector : public CustomTensorOperatorInstSelector {
protected:
    // from what i know now, i want to only inject into first fmul/fadd 
    // but i might also want to do both (deal with this after verifying IR and for other operators)

    virtual bool isInstFITarget(Instruction *inst) {
        return checkInCustomTensorOperator(inst, layerNo_WandB[0], layerName_WandB[0]);
    }

    virtual bool shouldInjectInstruction(Instruction *inst) {
        printf("TENSOR WEIGHT INJECTING \n");
        if (bias && inst->getOpcode() == Instruction::FAdd ||
            weights && inst->getOpcode() == Instruction::FMul) {
            // inCustomTensorOperator = false; // no need to keep looking in the current operator
            return true;
        }
        return false;
    }
    
public:
    WeightsAndBiasesInstSelector(){
        inCustomTensorOperator = false;
        injectInAll = false; 
    }

    virtual void getCompileTimeInfo(std::map<std::string, std::string> &info) {
        info["failure_class"] = "HardwareFault";
        info["failure_mode"] = "WeightsAndBiases";
        info["targets"] = "<instructions in main_graph() function and within \
            the specified tensor operator>";
        info["injector"] = "<fi_type>";
    }
};

static RegisterFIInstSelector X("WeightsAndBiases",
                                new WeightsAndBiasesInstSelector());
} //namespace llfi