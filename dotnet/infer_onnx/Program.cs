using FastBertTokenizer;
using CsBindgen;
using System.Text;

Console.WriteLine("Testing tokenizer...\n");

TestBertJapanese();
void TestBertJapanese()
{
    Console.WriteLine("Testing encoder(bert-base-japanese-v2)");

    var tok = new BertTokenizer();
    //tok.LoadFromHuggingFaceAsync("bert-base-uncased").GetAwaiter().GetResult();
    var vocab = File.OpenText("vocab.txt");
    tok.LoadVocabulary(vocab, false);
    //var tokenizerJson = File.OpenText("D:\\DATA\\tokenizer.json");
    //tok.LoadTokenizerJson(tokenizerJson.BaseStream);
    // Full-width symbols should be converted to half-width symbols
    // Example: ！ -> !
    // List of full-width symbols are: https://en.wikipedia.org/wiki/Halfwidth_and_fullwidth_forms
    var (inputIds, attentionMask, tokenTypeIds) = tok.Encode("こんにちは!");
    Console.WriteLine(string.Join(", ", inputIds.ToArray()));
    var decoded = tok.Decode(inputIds.Span);
    Console.WriteLine(decoded);
}

Console.WriteLine();

TestKoGPT2();
void TestKoGPT2()
{
    Console.WriteLine("Testing dcoder(kogpt2-base-v2)");
    unsafe
    {
        var data = new uint[] { 25906, 8702, 7801, 25856 };
        Console.WriteLine($"Input tokens: {string.Join(", ", data)}");
        fixed (uint* p = data)
        {
            var decoded = NativeMethods.tokenizer_decode(p, data.Length);
            try
            {
                var str = Encoding.UTF8.GetString(decoded->AsSpan());
                Console.WriteLine(str);
            }
            finally
            {
                NativeMethods.free_u8_string(decoded);
            }
        }
    }
}

Console.WriteLine("Done.");
