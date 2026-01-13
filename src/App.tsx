import { useState, useCallback, useEffect } from 'react';
import { useDropzone, type FileRejection } from 'react-dropzone';
import { Upload, X, FileImage, Loader2 } from 'lucide-react';
import axios from 'axios';

// Define the file structure with a preview URL
interface FileWithPreview extends File {
  preview: string;
}

const MAX_FILES = 20;
const MAX_SIZE_BYTES = 10 * 2 ** 20; // 10 MiB

export default function PaddleOCRFrontend() {
  const [files, setFiles] = useState<FileWithPreview[]>([]);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [tooLargeFiles, setTooLargeFiles] = useState<string[]>([]);
  const [invalidTypedFiles, setInvalidTypedFiles] = useState<string[]>([]);
  const axiosInstance = axios.create({ baseURL: import.meta.env.VITE_BACKEND_URI })
  const [isProcessing, setIsProcessing] = useState(false);
  const [taskUID, setTaskUID] = useState(null);
  const [progress, setProgress] = useState(0.0);
  const [results, setResults] = useState<Record<string, string>>({});

  // fetch token from backend
  useEffect(() => {
    const fetchToken = async () => {
      try {
        const res = await axiosInstance.get('/get-token');
        setToken(res.data.token);
      } catch (err) {
        console.error("Failed to fetch session token", err);
      }
    };
    fetchToken();
  }, []);

  // handle file drops and validation
  const onDrop = useCallback((acceptedFiles: File[], fileRejections: FileRejection[]) => {
    if (fileRejections.length > 0) {
      alert("Some files were rejected. Ensure they are images and under 10MB.");

      // ref: https://react-dropzone.js.org/
      // ref: https://stackoverflow.com/a/63753189/27092911
      const tempTooLargeFiles: string[] = [];
      const tempInvalidTypedFiles: string[] = [];
      fileRejections.map(({ file, errors }) => (
        errors.forEach((err) => {
          if (err.code === "file-invalid-type") {
            tempInvalidTypedFiles.push(`${file.path} (type: ${file.type})`);
          } else if (err.code === "file-too-large") {
            tempTooLargeFiles.push(
              `${file.path} (${Math.round(file.size / 2 ** 20 * 10) / 10} MiB)`
            );
          }
        })
      ));

      // postpone the state settings to prevent race conditions
      setTooLargeFiles(tempTooLargeFiles);
      setInvalidTypedFiles(tempInvalidTypedFiles);
    }

    const newFiles = acceptedFiles.slice(0, MAX_FILES - files.length).map(file =>
      Object.assign(file, {
        preview: URL.createObjectURL(file)
      })
    );

    setFiles(prev => [...prev, ...newFiles]);
  }, [files]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': [] },
    maxSize: MAX_SIZE_BYTES,
    maxFiles: MAX_FILES,
  });

  const removeFile = (name: string) => {
    setFiles(files.filter(f => f.name !== name));
  };

  const handleUpload = async () => {
    if (!token) return alert("No valid session token found.");
    setIsUploading(true);

    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    try {
      const res = await axiosInstance.post('/ocr', formData, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'multipart/form-data'
        }
      });
      setIsProcessing(true);
      setTaskUID(res.data.task_uid);
      console.log("Started processing.");
    } catch (err) {
      alert("Upload failed or server failed processing. Please reduce the resolution of images.");
      setIsProcessing(false);
    } finally {
      setIsUploading(false);
    }
  };

  // pull the processing status once every second
  // ref: https://stackoverflow.com/a/63143722/27092911
  useEffect(() => {
    const interval = setInterval(() => {
      if (isProcessing && taskUID) {
        const fetchResults = async () => {
          try {
            const res = await axiosInstance.get('/get-results/' + taskUID);
            setResults(res.data.results);
          }
          catch (err) {
            console.error("Failed to fetch the results", err);
          }
        }

        const fetchStatus = async () => {
          try {
            const res = await axiosInstance.get('/get-status/' + taskUID);
            setProgress(res.data.progress);
            if (res.data.status == "completed") {
              fetchResults();
              setIsProcessing(false);
              setTaskUID(null);
            }
          } catch (err) {
            console.error("Failed to fetch processing status", err);
          }
        };

        fetchStatus();
      }
    }, 1e3);
    return () => clearInterval(interval);
  }, [isProcessing, taskUID, progress])

  // cleanup object URLs to prevent memory leaks
  useEffect(() => {
    return () => files.forEach(file => URL.revokeObjectURL(file.preview));
  }, [files]);

  return (
    <div className="min-h-screen dark:bg-gray-900 p-8 font-sans">
      <div className="max-w-5xl mx-auto">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-gray-50">PaddleOCR Dashboard</h1>
          <p className="text-gray-300">Upload images for Markdown conversion (Max 20 files, 10MB each)</p>
        </header>

        {/* Upload Zone */}
        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors cursor-pointer dark:bg-gray-700
            ${isDragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-white hover:border-gray-400'}`}
        >
          <input {...getInputProps()} />
          <Upload className="mx-auto h-12 w-12 text-gray-50 mb-4" />
          <p className="text-lg text-gray-50">Drag & drop images here, or click to select</p>
        </div>

        <progress value={progress} max="1.0">{progress}%</progress>
        {Object.entries(results).length > 0 && (
          <div style={{whiteSpace: "pre-wrap"}}>
            <h2>Results:</h2>
            <ul>
            {Object.entries(results).map(([_, [filename, text]]) =>
              <li key={filename} className="text-gray-50" style={{marginBottom: "1em"}}>
                <b key={filename}>{filename}:</b><br/>
                {text}
              </li>
            )}
            </ul>
          </div>
        )}

        {/* Too large and/or invalid typed files */}
        {tooLargeFiles.length > 0 && (
          <div style={{ color: "red" }}>
            <p>Files too large:</p>
            <ul>
              {tooLargeFiles.map((line) => <li key={line} style={{ display: "list-item" }}>{line}</li>)}
            </ul>
          </div>
        )}

        {invalidTypedFiles.length > 0 && (
          <div style={{ color: "red" }}>
            Not images:
            <ul>
              {invalidTypedFiles.map((line) => <li key={line} style={{ display: "list-item" }}>{line}</li>)}
            </ul>
          </div>
        )}

        {/* Grid Preview */}
        {files.length > 0 && (
          <div className="mt-8 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {files.map((file) => (
              <div key={file.name} className="relative group bg-white p-2 rounded-lg shadow-sm border border-gray-200">
                <img
                  src={file.preview}
                  className="h-32 w-full object-cover rounded-md cursor-pointer"
                  onClick={() => setSelectedImage(file.preview)}
                />
                <button
                  onClick={() => removeFile(file.name)}
                  className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-1 shadow-lg opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  <X size={14} />
                </button>
                <div className="mt-2 flex items-center text-xs text-gray-500 truncate">
                  <FileImage size={12} className="mr-1" />
                  {file.name}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Action Button */}
        {files.length > 0 && (
          <div className="mt-8 flex justify-end">
            <button
              onClick={handleUpload}
              disabled={isUploading}
              className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg font-semibold flex items-center disabled:bg-blue-300"
            >
              {isUploading ? <Loader2 className="animate-spin mr-2" /> : null}
              Process {files.length} Files
            </button>
          </div>
        )}
      </div>

      {/* Modal for Original Preview */}
      {selectedImage && (
        <div
          className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4"
          onClick={() => setSelectedImage(null)}
        >
          <div className="relative max-w-4xl max-h-full">
            <img src={selectedImage} className="max-h-[90vh] rounded-lg shadow-2xl" />
            <button className="absolute top-4 right-4 text-white hover:text-gray-300">
              <X size={32} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}