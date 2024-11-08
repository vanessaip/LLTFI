#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"

#include "FIInstSelector.h"
#include "FICustomSelectorManager.h"

#include "CustomTensorOperatorInstSelector.h"

#include <vector>
using namespace llvm;

namespace llfi {

static cl::list< std::string > layerNo_weights("layerNo_weights", cl::desc("Layer Number in \
  which you want to inject bitflip faults. Pass 0 for injecting faults in all the \
  layers.\n Semi-colon seperated values. Example: 1;0;2"), cl::ZeroOrMore);

static cl::list< std::string > layerName_weights("layerName_weights", cl::desc("Layer Name in \
which you want to inject bitflip faults. Semi-colon seperated values. Example: \
Conv;Relu;Pool"), cl::ZeroOrMore);

class TensorWeightsInstSelector : public CustomTensorOperatorInstSelector {
protected:
    virtual bool isInstFITarget(Instruction *inst) {
        return checkInCustomTensorOperator(inst, layerNo_weights[0], layerName_weights[0]);
    }

    virtual bool shouldInjectInstruction(Instruction *inst) {
        printf("TENSOR WEIGHT INJECTING \n");
        if (inst->getOpcode() == Instruction::FMul) {
            inCustomTensorOperator = false; // no need to keep looking in the current operator
            return true;
        }
        return false;
    }
    
public:
    TensorWeightsInstSelector(){
        inCustomTensorOperator = false;
        injectInAll = false; 
    }

    virtual void getCompileTimeInfo(std::map<std::string, std::string> &info) {
        info["failure_class"] = "HardwareFault";
        info["failure_mode"] = "TensorWeights";
        info["targets"] = "<instructions in main_graph() function and within \
            the specified tensor operator>";
        info["injector"] = "<fi_type>";
    }
};

static RegisterFIInstSelector X("TensorWeights",
                                new TensorWeightsInstSelector());
} //namespace llfi