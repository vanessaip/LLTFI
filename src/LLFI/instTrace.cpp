#include <vector>

#include "llvm/Constants.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalValue.h"
#include "llvm/Pass.h"
#include "llvm/Function.h"
#include "llvm/Module.h"
#include "llvm/Instruction.h"
#include "llvm/Instructions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/InstIterator.h"
#include "llvm/Support/CommandLine.h"

#include "utils.h"

using namespace llvm;

cl::opt<std::string> OutputFilename("o", cl::desc("Specify output filename"), cl::value_desc("filename"));

namespace llfi {

struct instTrace : public FunctionPass {

static char ID;
Function::iterator lastBlock;
BasicBlock::iterator lastInst;
char *oFilename;
int oFilenameLength;

instTrace() : FunctionPass(ID) {}

virtual bool doInitialization(Module &M) {
	oFilenameLength = OutputFilename.size() + 1
	oFilename = new char[oFilenameLength];
	std::copy(OutputFilename.begin(), OutputFilename.end(), oFilename);
	oFilename[OutputFilename.size()] = '\0'; // don't forget the terminating 0
	return true;
}

virtual bool doFinalization(Module &) {
	delete[] oFilename;
	return true;
}

long fetchLLFIInstructionID(Instruction *targetInst) {
	return llfi::getLLFIIndexofInst(targetInst);
}

virtual bool runOnFunction(Function &F) {

	//Create handles to the functions parent module and context
	LLVMContext& context = F.getContext();
	Module *M = F.getParent();			

	//iterate through each basicblock of the function
	for (Function::iterator blockIterator = F.begin(), lastBlock = F.end();
		blockIterator != lastBlock; ++blockIterator) {

		BasicBlock* block = blockIterator;

		//Iterate through each instruction of the basicblock
		for (BasicBlock::iterator instIterator = blockIterator->begin(), 
			 lastInst = blockIterator->end(); 
			 instIterator != lastInst;
			 ++instIterator) {

			//Print some Debug Info as the pass is being run
			Instruction *inst = instIterator;

			errs() << "instTrace: Found Instruction\n";
			if (llfi::isLLFIInst(inst)) {
				errs() << "   Instruction was inserted/generated by LLFI\n";
			} else {
				errs() << "   Opcode Name: " << inst->getOpcodeName() << "\n"
				   	   << "   Opcode: " << inst->getOpcode() << "\n"
				       << "   Parent Function Name: " << inst->getParent()->getParent()->getNameStr() << "\n";
			}
			if (inst->getType() != Type::getVoidTy(context) && 
				!llfi::isLLFIInst(inst) && 
				inst != block->getTerminator()) {

				//Find instrumentation point for current instruction
				Instruction *insertPoint = llfi::getInsertPtrforRegsofInst(inst, inst);
				//insert an instruction Allocate stack memory to store/pass instruction value
				AllocaInst* ptrInst = new AllocaInst(inst->getType(), NULL, "", insertPoint);
				//Insert an instruction to Store the instruction Value!
				new StoreInst(inst, ptrInst, insertPoint);

				//Create the decleration of the printInstProfile Function
				std::vector<const Type*> parameterVector(4);
				parameterVector[0] = Type::getInt32Ty(context); //ID
				parameterVector[1] = Type::getInt32Ty(context); //OpCode
				parameterVector[2] = Type::getInt64Ty(context); //Size of Inst Value
				parameterVector[3] = ptrInst->getType();		//Ptr to Inst Value
				
				FunctionType* ppFuncType = FunctionType::get(Type::getVoidTy(context), parameterVector, 0 );
				Constant *ppFunc = M->getOrInsertFunction("printInstTracer", ppFuncType); 

				//Insert the tracing function, passing it the proper arguments
				std::vector<Value*> ppArgs;
				//Fetch the LLFI Instruction ID:
				ConstantInt* IDConstInt = ConstantInt::get(IntegerType::get(context,32), fetchLLFIInstructionID(inst));
				//Fetch the OPcode:
				ConstantInt* OPConstInt = ConstantInt::get(IntegerType::get(context,32), inst->getOpcode());
				//Fetch size of instruction value
				Constant* instValSize = ConstantExpr::getSizeOf(inst->getType());
				
				//Load All Arguments
				ppArgs.push_back(IDConstInt);
				ppArgs.push_back(OPConstInt);
				ppArgs.push_back(instValSize);
				ppArgs.push_back(ptrInst);

				//Create the Function
				CallInst::Create(ppFunc, ppArgs.begin(),ppArgs.end(), "", insertPoint);
			}
		}//Instruction Iteration
	}//BasicBlock Iteration

	return true; //Tell LLVM that the Function was modified
}//RunOnFunction
};//struct InstTrace

//Register the pass with the llvm
char instTrace::ID = 0;
static RegisterPass<instTrace> X("instTrace", "Traces instruction execution through program", false, false);

}//namespace llfi

