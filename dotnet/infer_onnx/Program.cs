using FastBertTokenizer;

Console.WriteLine("Hello, World!");

TestBertJapanese();

void TestBertJapanese()
{
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