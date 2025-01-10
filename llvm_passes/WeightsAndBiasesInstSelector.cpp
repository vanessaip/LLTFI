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
    uint32_t convFaddCount;

    virtual bool isInstFITarget(Instruction *inst) {
        return checkInCustomTensorOperator(inst, layerNo_WandB[0], layerName_WandB[0]);
    }

    virtual bool shouldInjectInstruction(Instruction *inst) {
        if (strcmp(currOperatorName.c_str(), "add") == 0){
            if (bias && inst->getOpcode() == Instruction::FAdd) {
                inCustomTensorOperator = false; // no need to keep looking in the current operator
                return true;
            }
            return false;
        } else if (strcmp(currOperatorName.c_str(), "matmul") == 0) {
            if (weights && inst->getOpcode() == Instruction::FMul) {
                inCustomTensorOperator = false;
                return true;
            }
            return false;
        } else if (strcmp(currOperatorName.c_str(), "conv") == 0
                    || strcmp(currOperatorName.c_str(), "gemm") == 0) // gemm has the same pattern as conv
            {
            // first (only) fmul
            if (weights && inst->getOpcode() == Instruction::FMul) {
                if (!bias) {
                    inCustomTensorOperator = false;
                }
                return true;
            // and second fadd
            } else if (bias && inst->getOpcode() == Instruction::FAdd) {
                assert(convFaddCount < 2);
                if (convFaddCount == 0) {
                    convFaddCount++;
                    printf("First fadd inst, skipping\n");
                    return false;
                } else if (convFaddCount == 1) {
                    convFaddCount = 0;
                    inCustomTensorOperator = false;
                    return true;
                } else {
                    printf("WeightsAndBiasesInstSelector: unexpected number of FAdd instructions: %d\n", convFaddCount);
                    return false;
                }
            }
            return false;
        } else {
            printf("WeightsAndBiasesInstSelector: unsupported operator type %s\n", currOperatorName.c_str());
            return false;
        }
        return false;
    }
    
public:
    WeightsAndBiasesInstSelector(){
        inCustomTensorOperator = false;
        injectInAll = false; 
        currOperatorName = "";
        convFaddCount = 0;  // for convolution operators, the bias terms are added in the second FAdd instance
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