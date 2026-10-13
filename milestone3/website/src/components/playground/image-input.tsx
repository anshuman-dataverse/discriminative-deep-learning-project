"use client";

import { Camera, ImageUp, X } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

export type LoadedImage = { src: string; image: HTMLImageElement | HTMLCanvasElement; width: number; height: number };

export function loadFromUrl(src: string): Promise<LoadedImage> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ src, image: img, width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => reject(new Error("Could not read that image"));
    img.src = src;
  });
}

export function ImageInput({ onImage, children }: { onImage: (img: LoadedImage) => void; children: React.ReactNode }) {
  const fileRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [camera, setCamera] = useState<MediaStream | null>(null);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fromFile = useCallback(
    async (file: File | undefined) => {
      if (!file || !file.type.startsWith("image/")) return;
      setError(null);
      try {
        onImage(await loadFromUrl(URL.createObjectURL(file)));
      } catch (e) {
        setError((e as Error).message);
      }
    },
    [onImage],
  );

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => fromFile(Array.from(e.clipboardData?.files ?? [])[0]);
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [fromFile]);

  useEffect(() => {
    if (videoRef.current && camera) videoRef.current.srcObject = camera;
    return () => camera?.getTracks().forEach((t) => t.stop());
  }, [camera]);

  const openCamera = async () => {
    setError(null);
    try {
      setCamera(await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: 1280 } }));
    } catch {
      setError("Camera not available (permission denied or no camera).");
    }
  };

  const capture = () => {
    const v = videoRef.current;
    if (!v) return;
    const c = document.createElement("canvas");
    c.width = v.videoWidth;
    c.height = v.videoHeight;
    c.getContext("2d")!.drawImage(v, 0, 0);
    onImage({ src: c.toDataURL("image/jpeg", 0.92), image: c, width: c.width, height: c.height });
    setCamera(null);
  };

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        fromFile(e.dataTransfer.files[0]);
      }}
      className={`relative rounded-xl transition-shadow ${dragging ? "ring-2 ring-series-1" : ""}`}
    >
      {children}
      {camera && (
        <div className="absolute inset-0 z-10 grid place-items-center overflow-hidden rounded-xl bg-black">
          <video ref={videoRef} autoPlay playsInline muted className="h-full w-full object-contain" />
          <div className="absolute bottom-3 flex gap-2">
            <button onClick={capture} className="rounded-full bg-white px-4 py-2 text-sm font-semibold text-black shadow">
              Capture
            </button>
            <button onClick={() => setCamera(null)} aria-label="Close camera" className="grid h-9 w-9 place-items-center rounded-full bg-white/20 text-white">
              <X className="h-4 w-4" />
            </button>
          </div>
        </div>
      )}
      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          onClick={() => fileRef.current?.click()}
          className="inline-flex items-center gap-2 rounded-lg bg-ink px-3 py-2 text-sm font-medium text-page hover:opacity-90"
        >
          <ImageUp className="h-4 w-4" /> Upload image
        </button>
        <button
          onClick={openCamera}
          className="inline-flex items-center gap-2 rounded-lg border border-line bg-surface px-3 py-2 text-sm font-medium hover:bg-surface-2"
        >
          <Camera className="h-4 w-4" /> Use camera
        </button>
        <span className="text-xs text-muted">or drop / paste an image</span>
        <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => fromFile(e.target.files?.[0])} />
      </div>
      {error && <p className="mt-2 text-xs text-critical">{error}</p>}
    </div>
  );
}
