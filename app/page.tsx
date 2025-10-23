'use client';

import { useEffect, useRef, useState } from 'react';
import { PoseLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [wingspan, setWingspan] = useState<number | null>(null);
  const [distance, setDistance] = useState<number | null>(null);
  const [poseLandmarker, setPoseLandmarker] = useState<PoseLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  
  // Average shoulder width in cm (typical adult range: 38-45cm)
  // This is used to calculate the user's distance from the camera
  const ASSUMED_SHOULDER_WIDTH_CM = 45;

  // Initialize MediaPipe PoseLandmarker
  useEffect(() => {
    const initializePoseLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );
        
        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          minPoseDetectionConfidence: 0.5,
          minPosePresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        
        setPoseLandmarker(landmarker);
      } catch (err) {
        console.error('Error initializing PoseLandmarker:', err);
        setError('Failed to initialize pose tracking.');
      }
    };

    initializePoseLandmarker();
  }, []);

  // Start camera
  useEffect(() => {
    const video = videoRef.current;
    
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
        });
        
        if (video) {
          video.srcObject = stream;
          video.addEventListener('loadeddata', () => {
            setIsLoading(false);
          });
        }
      } catch (err) {
        console.error('Error accessing camera:', err);
        setError('Failed to access camera. Please ensure camera permissions are granted.');
        setIsLoading(false);
      }
    };

    startCamera();

    // Cleanup function to stop the camera when component unmounts
    return () => {
      if (video && video.srcObject) {
        const stream = video.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, []);

  // Pose detection loop
  useEffect(() => {
    if (!poseLandmarker || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let lastVideoTime = -1;
    const startTimeMs = performance.now();

    const detectPose = async () => {
      if (!video || video.paused || video.ended) {
        animationFrameRef.current = requestAnimationFrame(detectPose);
        return;
      }

      // Set canvas size to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const currentTime = video.currentTime;
      if (currentTime !== lastVideoTime) {
        lastVideoTime = currentTime;

        // Calculate timestamp in milliseconds from start
        const timeMs = performance.now() - startTimeMs;
        
        // Detect pose
        const results = poseLandmarker.detectForVideo(video, timeMs);

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw pose landmarks
        if (results.landmarks && results.landmarks.length > 0) {
          const drawingUtils = new DrawingUtils(ctx);
          const landmarks = results.landmarks[0];

          // Draw pose connections
            drawingUtils.drawConnectors(
              landmarks,
            PoseLandmarker.POSE_CONNECTIONS,
            { color: '#00FF00', lineWidth: 4 }
            );
            drawingUtils.drawLandmarks(landmarks, {
              color: '#FF0000',
              lineWidth: 2,
            radius: 4
          });

          // MediaPipe Pose landmark indices:
          // 11 = Left Shoulder, 12 = Right Shoulder
          // 15 = Left Wrist, 16 = Right Wrist
          const leftShoulder = landmarks[11];
          const rightShoulder = landmarks[12];
          const leftWrist = landmarks[15];
          const rightWrist = landmarks[16];

          // Calculate shoulder width in pixels
          const shoulderDx = (rightShoulder.x - leftShoulder.x) * canvas.width;
          const shoulderDy = (rightShoulder.y - leftShoulder.y) * canvas.height;
          const shoulderWidthPixels = Math.sqrt(shoulderDx * shoulderDx + shoulderDy * shoulderDy);

          // Calculate camera distance using the known shoulder width
            // Typical webcam has ~70° horizontal FOV
            const fovDegrees = 70;
          
          // Using similar triangles:
          // real_width / distance = pixel_width / focal_length
          // focal_length (in pixels) = canvas_width / (2 * tan(fov/2))
          const focalLengthPixels = canvas.width / (2 * Math.tan((fovDegrees / 2) * (Math.PI / 180)));
          
          // Calculate distance: distance = (real_width * focal_length) / pixel_width
          const calculatedDistance = (ASSUMED_SHOULDER_WIDTH_CM * focalLengthPixels) / shoulderWidthPixels;
          setDistance(calculatedDistance);

          // Calculate wingspan using the calculated distance
          const wristDx = (rightWrist.x - leftWrist.x) * canvas.width;
          const wristDy = (rightWrist.y - leftWrist.y) * canvas.height;
          const wingspanPixels = Math.sqrt(wristDx * wristDx + wristDy * wristDy);
          
          // Convert wingspan pixels to cm using the calculated distance
          const wingspanCm = (wingspanPixels * calculatedDistance) / focalLengthPixels;
            setWingspan(wingspanCm);

            // Draw line between wrists to show wingspan
            ctx.beginPath();
          ctx.moveTo(leftWrist.x * canvas.width, leftWrist.y * canvas.height);
          ctx.lineTo(rightWrist.x * canvas.width, rightWrist.y * canvas.height);
            ctx.strokeStyle = '#FFFF00';
          ctx.lineWidth = 5;
            ctx.stroke();
            
            // Draw circles at wrists
            ctx.fillStyle = '#FFFF00';
            ctx.beginPath();
          ctx.arc(leftWrist.x * canvas.width, leftWrist.y * canvas.height, 10, 0, 2 * Math.PI);
            ctx.fill();
            ctx.beginPath();
          ctx.arc(rightWrist.x * canvas.width, rightWrist.y * canvas.height, 10, 0, 2 * Math.PI);
            ctx.fill();

          // Draw line between shoulders for reference
          ctx.beginPath();
          ctx.moveTo(leftShoulder.x * canvas.width, leftShoulder.y * canvas.height);
          ctx.lineTo(rightShoulder.x * canvas.width, rightShoulder.y * canvas.height);
          ctx.strokeStyle = '#00FFFF';
          ctx.lineWidth = 3;
          ctx.stroke();
          } else {
            setWingspan(null);
          setDistance(null);
        }
      }

      animationFrameRef.current = requestAnimationFrame(detectPose);
    };

    detectPose();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [poseLandmarker, ASSUMED_SHOULDER_WIDTH_CM]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gray-900 p-4">
      <div className="w-full max-w-5xl">
        <h1 className="text-4xl font-bold text-white text-center mb-8">
          MediaPipe Wingspan Tracker
        </h1>
        
        <div className="text-center mb-4 text-sm text-blue-300 bg-blue-900/30 rounded-lg p-3">
          📏 Face the camera and extend both arms horizontally. Distance is calculated automatically!
        </div>
        
        <div className="relative bg-black rounded-lg overflow-hidden shadow-2xl">
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-800 z-10">
              <div className="text-white text-xl">Loading camera and pose tracking model...</div>
            </div>
          )}
          
          {error && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-800 z-10">
              <div className="text-red-400 text-xl text-center px-4">{error}</div>
            </div>
          )}
          
          <video
            ref={videoRef}
            autoPlay
            playsInline
            className="w-full h-auto"
          />
          
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full"
          />
        </div>

        <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="p-6 bg-gray-800 rounded-lg">
            <div className="text-center">
              <h2 className="text-xl font-semibold text-white mb-3">Distance from Camera</h2>
              {distance !== null ? (
                <div>
                  <div className="text-4xl font-bold text-cyan-400 mb-1">
                    {distance.toFixed(0)} cm
                  </div>
                  <div className="text-lg text-cyan-300">
                    {(distance / 100).toFixed(2)} meters
                  </div>
                </div>
              ) : (
                <div className="text-lg text-gray-400">
                  Face the camera to calculate distance
                </div>
              )}
            </div>
          </div>

          <div className="p-6 bg-gray-800 rounded-lg">
            <div className="text-center">
              <h2 className="text-xl font-semibold text-white mb-3">Your Wingspan</h2>
              {wingspan !== null ? (
                <div>
                  <div className="text-4xl font-bold text-green-400 mb-1">
                    {wingspan.toFixed(1)} cm
                  </div>
                  <div className="text-lg text-green-300">
                    {(wingspan / 100).toFixed(2)} meters
                  </div>
                </div>
              ) : (
                <div className="text-lg text-gray-400">
                  Extend both arms horizontally
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
