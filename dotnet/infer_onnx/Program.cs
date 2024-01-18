using infer_onnx;

TestFunctions tf = new();

Console.WriteLine("Testing tokenizer...\n");
tf.TestBertJapanese("打ち合わせが終わった後にご飯を食べましょう。");

Console.WriteLine();
tf.TestKoGPT2();

Console.WriteLine("Done tokenizer tests.");
Console.WriteLine("Testing ONNX...\n");

tf.TestONNX();



