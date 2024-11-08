#include "llvm/IR/Instructions.h"
#include "llvm/Support/CommandLine.h"

#include "FIInstSelector.h"

#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <unordered_map>
#include <cassert>

using namespace llvm;

/* Common definitions used to inject within Tensor Operators */

namespace llfi {

enum OperatorStatus {
    op_start = 1,
    op_end = 2,
};

// Return an array our of string of comma-seperated values.
static std::vector<std::string> getCommaSeperateVals(std::string inp) {

    std::string s = inp;
    std::string delimiter = ";";
    std::vector<std::string> retval;
    size_t pos = 0;

    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        retval.push_back(token);
        s.erase(0, pos + delimiter.length());
    }

    if ((pos = s.find(delimiter)) == std::string::npos) {
        retval.push_back(s);
    }

    return retval;
}

// Data structure to keep track of every Tensor Operator.
struct Operator {

    std::string OperatorName;
    // ONNX mlir assigns a unique ID to every operator.
    int64_t OperatorNumber;
    // Number of times we have seen this operator.
    int OperatorCount;
    // Operator number to do FI
    int FIOperatorCount;

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

    Operator(std::string name, std::string count) {

        OperatorName = name;
        FIOperatorCount = atoll(count.c_str());
        OperatorCount = 0;
        OperatorNumber = getOperatorNumber(name);

        if (OperatorNumber == -1) {
            std::cout<<"Operator name "<< OperatorName.c_str() <<
                " not found.\n";
            std::cout<<"Please use the following operator name(s):\
            conv, relu, maxpool, matmul, add, avgpool, all, and softmax.";
            assert(false && "Invalid input operator name");
        }

        assert(FIOperatorCount >= 0 && "Invalid input FI operator number");
    }

    bool doFaultInjection(){

        OperatorCount++;

        // Inject fault in the user-specified operator count.
        if (FIOperatorCount == 0 || FIOperatorCount == OperatorCount)
            return true;

        return false;
    }
}; // End of struct Operator.

class CustomTensorOperatorInstSelector : public HardwareFIInstSelector {
  protected:
    bool inCustomTensorOperator;
    std::unordered_map<int64_t, std::vector<Operator*>> map;
    bool injectInAll;

    bool checkInCustomTensorOperator(llvm::Instruction *inst, std::string layerNo, std::string layerName);

    // Add Metadata to LLVM instructions; Only for debugging purposes!
    void addMetadata(llvm::Instruction *ins, char *st = NULL){
      LLVMContext& C = ins->getContext();
      MDNode* N = MDNode::get(C, MDString::get(C, (!st) ? "t" : st));
      ins->setMetadata("Debug", N);
    }

    // Initializes Layer name and number
    void initializeLayerNameAndNumber(std::string layerNo,
            std::string layerName) {

      std::vector<std::string> OperatorNames =
          getCommaSeperateVals(layerName);
      std::vector<std::string> OperatorNumbers =
          getCommaSeperateVals(layerNo);

      assert(OperatorNumbers.size() == OperatorNames.size() &&
          "Number of CSVs given to the layerNo and layerName should be equal");

      for (int i  = 0; i < OperatorNames.size(); i++) {

          std::string name = OperatorNames[i];
          std::string number = OperatorNumbers[i];

          // Inject in all operators.
          if (strcmp(name.c_str(), "all") == 0 ||
              strcmp(name.c_str(), "All") == 0) {
              injectInAll = true;
              break;
          }

          int64_t code = Operator::getOperatorNumber(name);

          // if this operator is already in the map
          if (map.find(code) != map.end()) {

              Operator *temp = new Operator(name, number);
              map[code].push_back(temp);
          }
          else {

              std::vector<Operator*> OpArr;
              Operator *temp = new Operator(name, number);
              OpArr.push_back(temp);
              map.insert(make_pair(code, OpArr));
          }
      }
    }

    bool shouldInjectFaultInOperator(int64_t number) {

      if (injectInAll) return true;

      // If the operator isn't present in the map.
      if (map.find(number) == map.end()) return false;
      else {

          std::vector<Operator*> temp = map[number];
          bool result;

          for (auto it : temp) {
              result |= it->doFaultInjection();
          }

          return result;
      }
    }
  private:
    virtual bool isInstFITarget(Instruction *inst);

    virtual bool shouldInjectInstruction(Instruction *inst);
    
  public:
    CustomTensorOperatorInstSelector();

    virtual void getCompileTimeInfo(std::map<std::string, std::string> &info);
};

}
