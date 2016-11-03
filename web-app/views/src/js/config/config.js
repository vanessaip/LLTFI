var softwareInjectionTypeOptions = [
	{value: "CPUHog(Res)", text: "CPUHog(Res)"},
	{value: "DataCorruption(Data)", text: "DataCorruption(Data)"},
	{value: "HighFrequentEvent(Timing)", text: "HighFrequentEvent(Timing)"},
	{value: "IncorrectOutput(API)", text: "IncorrectOutput(API)"},
	{value: "NoOutput(API)", text: "NoOutput(API)"}
];

// Hardware injection types from gui/gui_config.yaml
var hardwareInjectionTypeOptions = [
	{value: "ret", text: "ret-(ReturnInst)"},
	{value: "br", text: "br-(BranchInst)"},
	{value: "switch", text: "switch-(SwitchInst)"},
	{value: "indirectbr", text: "indirectbr-(IndirectBrInst)"},
	{value: "invoke", text: "invoke-(InvokeInst)"},
	{value: "resume", text: "resume"},
	{value: "unreachable", text: "unreachable-(UnreachableInst)"},
	{value: "add", text: "add-(BinaryADD)"},
	{value: "fadd", text: "fadd"},
	{value: "sub", text: "sub-(BinarySUB)"},
	{value: "fsub", text: "fsub"},
	{value: "mul", text: "mul-(BinaryMUL)"},
	{value: "fmul", text: "fmul"},
	{value: "udiv", text: "udiv"},
	{value: "sdiv", text: "sdiv"},
	{value: "fdiv", text: "fdiv"},
	{value: "urem", text: "urem"},
	{value: "srem", text: "srem"},
	{value: "frem", text: "frem"},
	{value: "shl", text: "shl"},
	{value: "lshr", text: "lshr"},
	{value: "ashr", text: "ashr"},
	{value: "and", text: "and-(BinaryOR)"},
	{value: "or", text: "or-(BinaryAND)"},
	{value: "xor", text: "xor-(BinaryXOR)"},
	{value: "extractelement", text: "extractelement"},
	{value: "insertelement", text: "insertelement"},
	{value: "shufflevector", text: "shufflevector"},
	{value: "extractvalue", text: "extractvalue"},
	{value: "insertvalue", text: "insertvalue"},
	{value: "alloca", text: "alloca"},
	{value: "load", text: "load-(LoadInst)"},
	{value: "store", text: "store-(StoreInst)"},
	{value: "fence", text: "fence"},
	{value: "cmpxchg", text: "cmpxchg"},
	{value: "atomicrmw", text: "atomicrmw"},
	{value: "getelementptr", text: "getelementptr"},
	{value: "trunc", text: "trunc"},
	{value: "zext", text: "zext"},
	{value: "fptrunc", text: "fptrunc"},
	{value: "fpext", text: "fpext"},
	{value: "fptoui", text: "fptoui"},
	{value: "fptosi", text: "fptosi"},
	{value: "uitofp", text: "uitofp"},
	{value: "sitofp", text: "sitofp"},
	{value: "ptrtoint", text: "ptrtoint"},
	{value: "inttoptr", text: "inttoptr"},
	{value: "bitcast", text: "bitcast"},
	{value: "addrspacecast", text: "addrspacecast"},
	{value: "icmp", text: "icmp"},
	{value: "fcmp", text: "fcmp"},
	{value: "phi", text: "phi"},
	{value: "select", text: "select"},
	{value: "call", text: "call"},
	{value: "va_arg", text: "va_arg"},
	{value: "landingpad", text: "landingpad"}
];

export const injectionType = {
	softwareInjectionTypeOptions: softwareInjectionTypeOptions,
	hardwareInjectionTypeOptions: hardwareInjectionTypeOptions
};