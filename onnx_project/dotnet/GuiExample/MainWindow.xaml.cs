using Microsoft.Win32;
using System.Windows;
using Downloader;
using System.Net;
using System.ComponentModel;
using System.IO;
using System.Security.Cryptography;
using SevenZip;

namespace GuiExample
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly DownloadService downloader = new(new DownloadConfiguration
        {
            RequestConfiguration =
            {
                Proxy = WebRequest.GetSystemWebProxy()
            }
        });

        private const string MODEL_URL = "https://github.com/sappho192/ffxiv-ja-ko-translator/releases/download/0.2.1/onnx_model.7z";
        private const string HASH = "607646b69cd162b6d575df9b14537183";
        private const string MODEL_FILE_NAME = "onnx_model.7z";

        public MainWindow()
        {
            InitializeComponent();
            InitializeModelDirPath();
            InitDownloader();
            SevenZip.SevenZipBase.SetLibraryPath(
                Path.Combine(Directory.GetCurrentDirectory(), "7z.dll"));
        }

        private void InitDownloader()
        {
            downloader.DownloadStarted += Downloader_DownloadStarted;
            downloader.DownloadProgressChanged += Downloader_DownloadProgressChanged;
            downloader.DownloadFileCompleted += Downloader_DownloadFileCompleted;
        }

        private void Downloader_DownloadFileCompleted(object? sender, AsyncCompletedEventArgs e)
        {
            var modelDirPath = string.Empty;
            Application.Current.Dispatcher.Invoke(() =>
            {
                btDownloadModel.IsEnabled = true;
                modelDirPath = tbModelDirPath.Text;
            });

            string filePath = $"{modelDirPath}\\{MODEL_FILE_NAME}";
            if (!CheckHash(filePath, HASH))
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    sbSnackbar.Show("Download failed. Please try again.");
                });
                return;
            }
            else
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    sbSnackbar.Show("Download completed. Extracting the model file...");
                });
                ExtractModelArchive();
            }
        }

        private static bool CheckHash(string filePath, string hash)
        {
            using var md5 = MD5.Create();
            using var stream = File.OpenRead(filePath);
            var hashBytes = md5.ComputeHash(stream);
            var hashStr = BitConverter.ToString(hashBytes).Replace("-", "").ToLowerInvariant();
            return hashStr == hash;
        }

        private void ExtractModelArchive()
        {
            string modelDirPath = "";
            Application.Current.Dispatcher.Invoke(() =>
            {
                modelDirPath = tbModelDirPath.Text;
            });

            var filePath = $"{modelDirPath}\\{MODEL_FILE_NAME}";
            var extractor = new SevenZipExtractor(filePath);
            extractor.ExtractionFinished += (sender, args) =>
            {
                Application.Current.Dispatcher.Invoke(() =>
                {
                    sbSnackbar.Show("Extracted the model file.");
                });
            };
            extractor.ExtractArchive(modelDirPath);

            //using var archive = SevenZipArchive.Open(filePath);
            //foreach (var entry in archive.Entries.Where(entry => !entry.IsDirectory))
            //{
            //    entry.WriteToDirectory(modelDirPath, new ExtractionOptions()
            //    {
            //        ExtractFullPath = true,
            //        Overwrite = true
            //    });
            //}
        }

        private void Downloader_DownloadProgressChanged(object? sender, Downloader.DownloadProgressChangedEventArgs e)
        {
            //Application.Current.Dispatcher.Invoke(() =>
            //{
            //    lbDownloadProgress.Content = $"{e.ProgressPercentage:F2}%";
            //    lbSpeedProgress.Content = $"{e.AverageBytesPerSecondSpeed}B/s";
            //});
        }

        private void Downloader_DownloadStarted(object? sender, DownloadStartedEventArgs e)
        {
            sbSnackbar.Show("Downloading the model file...");
        }

        private void InitializeModelDirPath()
        {
            // Get current absolute directory path
            string currentDirPath = Directory.GetCurrentDirectory();
            // Make Onnx model directory path
            string onnxDirPath = Path.Combine(currentDirPath, "onnx");
            tbModelDirPath.Text = onnxDirPath;
        }

        private void btLoadModel_Click(object sender, RoutedEventArgs e)
        {
            (var allFileExist, var message) = checkModelFiles();
            if (!allFileExist)
            {// "Please download the model first."
                sbSnackbar.Show(message);
            }
            else
            {
                // Load model

                sbSnackbar.Show("Model loaded.");
            }
        }

        private async void btDownloadModel_Click(object sender, RoutedEventArgs e)
        {
            btDownloadModel.IsEnabled = false;

            await DownloadModel();
        }

        private async Task DownloadModel()
        {
            var modelDirPath = tbModelDirPath.Text;
            var modelArchivePath = $"{modelDirPath}\\{MODEL_FILE_NAME}";
            if (File.Exists(modelArchivePath))
            {
                File.Delete(modelArchivePath);
            }

            // Get current directoryinfo
            var directoryInfo = new DirectoryInfo(modelDirPath);
            await downloader.DownloadFileTaskAsync(MODEL_URL, directoryInfo);
        }

        private void btInputText1_Click(object sender, RoutedEventArgs e)
        {
            tbSrcText.Text = "ギルガメッシュ討伐戦に行ってきます。一緒に行きましょうか？";
            tbDstText.Text = "";
        }

        private void btInputText2_Click(object sender, RoutedEventArgs e)
        {
            tbSrcText.Text = "絶アルテマウェポン破壊作戦をクリアした事ありますか？";
            tbDstText.Text = "";
        }

        private void btInputText3_Click(object sender, RoutedEventArgs e)
        {
            tbSrcText.Text = "美容師の呼び鈴を使って髪方を変えますよ。";
            tbDstText.Text = "";
        }

        private void btTranslate_Click(object sender, RoutedEventArgs e)
        {
            //sbSnackbar.Show();
        }

        private (bool, string) checkModelFiles()
        {
            // Check onnx directory
            string onnxDirPath = tbModelDirPath.Text;
            if (!Directory.Exists(onnxDirPath))
            {
                return (false, "The path doesn't exist.");
            }
            // Check onnx model files
            string[] onnxModelFiles = [
                "encoder_model.onnx",
                "decoder_model_merged.onnx",
                //"tokenizer.json", 
                //"vocab.txt" 
            ];
            foreach (string onnxModelFile in onnxModelFiles)
            {
                string onnxModelFilePath = Path.Combine(onnxDirPath, onnxModelFile);
                if (!File.Exists(onnxModelFilePath))
                {
                    return (false, $"Path doesn't exist: {onnxModelFilePath}");
                }
            }

            return (true, string.Empty);
        }

        private void btSearchModelDirPath_Click(object sender, RoutedEventArgs e)
        {
            var dialog = new OpenFolderDialog
            {
                Multiselect = false,
                Title = "Select ONNX model directory",
                InitialDirectory = tbModelDirPath.Text
            };
            if (dialog.ShowDialog() == true)
            {
                tbModelDirPath.Text = dialog.FolderName;
            }
        }

        private void btExtractModel_Click(object sender, RoutedEventArgs e)
        {
            ExtractModelArchive();
        }

        private void btHashCheck_Click(object sender, RoutedEventArgs e)
        {
            var modelDirPath = tbModelDirPath.Text;
            string filePath = $"{modelDirPath}\\onnx_model.7z";
            if (!CheckHash(filePath, HASH))
            {
                sbSnackbar.Show("Hash check failed. Please try again.");
                return;
            }
            else
            {
                sbSnackbar.Show("Hash check completed.");

            }
        }
    }
}