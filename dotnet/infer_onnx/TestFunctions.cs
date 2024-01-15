using CsBindgen;
using FastBertTokenizer;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using NumSharp;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace infer_onnx
{
    public class TestFunctions
    {
        private readonly BertTokenizer srcTokenizer = new();
        private readonly KoGPT2Tokenizer trgTokenizer = new();

        public TestFunctions()
        {
            //tok.LoadFromHuggingFaceAsync("bert-base-uncased").GetAwaiter().GetResult();
            var vocab = File.OpenText("vocab.txt");
            srcTokenizer.LoadVocabulary(vocab, false);
        }

        public void TestBertJapanese(string text)
        {
            Console.WriteLine("Testing encoder(bert-base-japanese-v2)");
            
            //var tokenizerJson = File.OpenText("D:\\DATA\\tokenizer.json");
            //tok.LoadTokenizerJson(tokenizerJson.BaseStream);
            // Full-width symbols should be converted to half-width symbols
            // Example: ！ -> !
            // List of full-width symbols are: https://en.wikipedia.org/wiki/Halfwidth_and_fullwidth_forms
            var (inputIds, attentionMask, tokenTypeIds) = srcTokenizer.Encode(text);
            Console.WriteLine(string.Join(", ", inputIds.ToArray()));
            var decoded = srcTokenizer.Decode(inputIds.Span);
            Console.WriteLine(decoded);
        }

        public void TestKoGPT2()
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

        public void TestONNX()
        {
            const string encoderPath = @"D:\REPO\onnx-hf-test\python\onnx\encoder_model.onnx";
            const string decoderPath = @"D:\REPO\onnx-hf-test\python\onnx\decoder_model_merged.onnx";
            var sessionOptions = new SessionOptions { 
                LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR
            };
            var encoderSession = new InferenceSession(encoderPath, sessionOptions);
            var decoderSession = new InferenceSession(decoderPath, sessionOptions);

            var inputText = "逃げろ!";
            Console.WriteLine($"Input text: {inputText}");
            var (inputIds, attentionMask, tokenTypeIds) = srcTokenizer.Encode(inputText);
            // inputIds to NDArray
            NDArray ndInputIds = np.array(inputIds.ToArray());
            ndInputIds = np.expand_dims(ndInputIds, 0);
            NDArray ndAttentionMask = np.array(attentionMask.ToArray());
            ndAttentionMask = np.expand_dims(ndAttentionMask, 0);

            var encoderInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", ndInputIds.ToMuliDimArray<long>().ToTensor<long>()),
                NamedOnnxValue.CreateFromTensor("attention_mask", ndAttentionMask.ToMuliDimArray<long>().ToTensor<long>())
            };

            var encoderResults = encoderSession.Run(encoderInput);
            var encoderResult = encoderResults.First();

            var singleBoolArray = new bool[] { false };
            var useCacheBranch = np.array(singleBoolArray);
            //useCacheBranch = np.expand_dims(useCacheBranch, 0);

            //var zeros = np.zeros<float>(1, 12, inputIds.Length, 64);

            var decoderInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input_ids", ndInputIds.ToMuliDimArray<long>().ToTensor<long>()),
                NamedOnnxValue.CreateFromTensor("encoder_hidden_states", encoderResult.AsTensor<float>()),
                NamedOnnxValue.CreateFromTensor("use_cache_branch", useCacheBranch.ToMuliDimArray<bool>().ToTensor<bool>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.0.key", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.0.value", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.1.key", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.1.value", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.2.key", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.2.value", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.3.key", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.3.value", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.4.key", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.4.value", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.5.key", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.5.value", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.6.key", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.6.value", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.7.key", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.7.value", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.8.key", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.8.value", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.9.key", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.9.value", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.10.key", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.10.value", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.11.key", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
                NamedOnnxValue.CreateFromTensor("past_key_values.11.value", np.zeros<float>(1, 12, inputIds.Length, 64).ToMuliDimArray<float>().ToTensor<float>()),
            };

            var generatedText = GreedySearch(decoderInput, encoderSession, decoderSession);
            Console.WriteLine($"Translated text: {generatedText}");
        }

        private string GreedySearch(List<NamedOnnxValue> decoderInput, InferenceSession encoderSession, InferenceSession decoderSession, int maxLength = 50)
        {
            // Initialize the input for the decoder with the BOS token ID
            long bosTokenId = 1;
            long eosTokenId = 1;
            var inputIds = new List<long> { bosTokenId };
            var inputIdsTensor = new DenseTensor<long>(inputIds.ToArray(), new[] { 1, 1 });
            // Update input_ids for the decoder
            decoderInput[0] = NamedOnnxValue.CreateFromTensor("input_ids", inputIdsTensor);

            // Initialize the list to store the generated tokens
            List<long> generatedTokens = new List<long>();

            // Greedy search loop
            for (int i = 0; i < maxLength; i++)
            {
                // Run the decoder model
                using var decoderResults = decoderSession.Run(decoderInput);

                // Update past_key_values with the current output
                if (decoderResults[1].Value != null)
                {
                    // Update the past_key_values with the latest decoder output: from 0 to 11
                    decoderInput.Find(input => input.Name.Equals("past_key_values.0.key")).Value = decoderResults[1].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.0.value")).Value = decoderResults[2].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.1.key")).Value = decoderResults[3].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.1.value")).Value = decoderResults[4].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.2.key")).Value = decoderResults[5].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.2.value")).Value = decoderResults[6].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.3.key")).Value = decoderResults[7].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.3.value")).Value = decoderResults[8].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.4.key")).Value = decoderResults[9].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.4.value")).Value = decoderResults[10].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.5.key")).Value = decoderResults[11].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.5.value")).Value = decoderResults[12].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.6.key")).Value = decoderResults[13].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.6.value")).Value = decoderResults[14].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.7.key")).Value = decoderResults[15].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.7.value")).Value = decoderResults[16].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.8.key")).Value = decoderResults[17].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.8.value")).Value = decoderResults[18].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.9.key")).Value = decoderResults[19].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.9.value")).Value = decoderResults[20].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.10.key")).Value = decoderResults[21].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.10.value")).Value = decoderResults[22].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.11.key")).Value = decoderResults[23].Value;
                    decoderInput.Find(input => input.Name.Equals("past_key_values.11.value")).Value = decoderResults[24].Value;
                }

                // Get the logits from the decoder results
                var logits = decoderResults.First().AsTensor<float>();

                // Apply softmax to logits to get probabilities
                var probabilities = Softmax(logits);
                var nextTokenId = ArgMax(probabilities.ToTensor());

                // Append the token to the list
                generatedTokens.Add(nextTokenId);

                // Prepare the input for the next iteration
                inputIdsTensor = new DenseTensor<long>(new long[] { nextTokenId }, new[] { 1, 1 });

                // Check if EOS token is generated
                if (nextTokenId == eosTokenId)
                {
                    break;
                }
            }

            // Decode the generated tokens into text
            var generatedText = trgTokenizer.Decode(generatedTokens.Select(item => (uint)item).ToArray());
            return generatedText;
        }

        private float[] Softmax(Tensor<float> logits)
        {
            // Find the maximum value in the logits
            float maxLogit = logits.Max();

            // Subtract the max value from each logit to avoid overflow during the exponential operation
            float[] expLogits = logits.Select(logit => (float)Math.Exp(logit - maxLogit)).ToArray();

            // Calculate the sum of the exponentials
            float sumExpLogits = expLogits.Sum();

            // Divide each exponential by the sum to get the softmax probabilities
            float[] softmaxProbabilities = expLogits.Select(expLogit => expLogit / sumExpLogits).ToArray();

            return softmaxProbabilities;
        }

        private int ArgMax(Tensor<float> probabilities)
        {
            // Find the index of the maximum value in the probabilities
            int argMaxIndex = Array.IndexOf(probabilities.ToArray(), probabilities.Max());

            return argMaxIndex;
        }

    }
}
