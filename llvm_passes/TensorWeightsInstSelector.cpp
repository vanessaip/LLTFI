#include "llvm/IR/Instructions.h"
#include "llvm/IR/Constants.h"

#include "FIInstSelector.h"
#include "FICustomSelectorManager.h"

#include "MLInstSelectors.h"

#include <vector>
using namespace llvm;

namespace llfi {

enum OperatorStatus {
    op_start = 1,
    op_end = 2,
};

class TensorWeightsInstSelector : public HardwareFIInstSelector {
public:
    // Get unique Id corresponding to the ONNX operator.
    static int64_t getOperatorNumber(std::string name) {

        char opname[100];
        std::transform(name.begin(), name.end(), name.begin(),
                        [](unsigned char c){ return std::tolower(c); });

        strcpy(opname, name.c_str());

        std::cout<<"OperatorName: "<<opname<<"\n";

        // ONNX assigns unique IDs to each tensor operator.
        std::map<std::string, int64_t> ONNXOperatorId = {
            {"conv", 1986948931},
            {"relu", 1970038098},
            {"maxpool", 30521821366870349},
            {"matmul", 119251066446157},
            {"add", 6579265},
            {"avgpool", 30521821365761601},
            {"softmax", 33884119937478483},
            {"loop", 1886351180},
            {"nonmaxs", 23494782373228366},
            {"unsqueeze", 28540527736745557}
        };

        if (ONNXOperatorId.find(opname) == ONNXOperatorId.end())
            return -1;

        return ONNXOperatorId[opname];
    }
private:
    bool inCustomTensorOperator;
    // CustomTensorOperatorInstSelector::Operator op;
    // std::unordered_map<int64_t, std::vector<Operator*>> operator_map;
    // bool injectInAll;

    // Add Metadata to LLVM instructions; Only for debugging purposes!
    void addMetadata(llvm::Instruction *ins, char *st = NULL){
         LLVMContext& C = ins->getContext();
         MDNode* N = MDNode::get(C, MDString::get(C, (!st) ? "t" : st));
         ins->setMetadata("Debug", N);
    }

    bool checkOperatorType(uint64_t onnx_code) {
        // placeholder logic now
        return getOperatorNumber("matmul") == onnx_code;
    }

    virtual bool isInstFITarget(Instruction *inst) {
        // check we're in the main graph
        if (inst->getParent()->getParent()->getName() == "main_graph") {

            // if it's a call, check if we're calling an OMInstrument function
            if (inst->getOpcode() == Instruction::Call) {
                // check if call instruction is for tensor operator
                CallInst* callinst = dyn_cast<CallInst>(inst);

                // If this is OMInstrument function?
                if ((callinst->getCalledFunction())->getName() == "OMInstrumentPoint") {
                    Value* arg1 = callinst->getArgOperand(0);
                    Value* arg2 = callinst->getArgOperand(1);

                    ConstantInt* ci1 = dyn_cast<ConstantInt>(arg1);
                    ConstantInt* ci2 = dyn_cast<ConstantInt>(arg2);

                    int64_t argValue1 = ci1->getSExtValue();
                    int64_t argValue2 = ci2->getSExtValue();

                    //check that we're inside the operator and if it's matmul
                    if (argValue2 == op_start && checkOperatorType(argValue1)) {
                        inCustomTensorOperator = true; // we are about to enter the operator code
                    } else if (argValue2 == op_end) {
                        inCustomTensorOperator = false; // reached the end of the operator
                    }
                }
            }
            // find the first FMul
            if (inCustomTensorOperator && inst->getOpcode() == Instruction::FMul) {
                addMetadata(inst, "Injected fault");
                return true; // we're just gonna let the user input choose the register for now
            }
            return false;
        }
        return false;
    }
    
public:
    TensorWeightsInstSelector(){
        inCustomTensorOperator = false;
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