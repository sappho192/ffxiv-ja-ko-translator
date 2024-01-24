fn main() {
    csbindgen::Builder::default()
        .input_extern_file("src/lib.rs")
        .csharp_dll_name("decoder")
        .csharp_class_name("NativeMethods")
        .generate_csharp_file("../../dotnet/infer_onnx/NativeMethods.cs")
        .unwrap();
}
