'use client';

import { useEffect, useRef, useState } from 'react';
import { HandLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [distance, setDistance] = useState<number | null>(null);
  const [handLandmarker, setHandLandmarker] = useState<HandLandmarker | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Initialize MediaPipe HandLandmarker
  useEffect(() => {
    const initializeHandLandmarker = async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );
        
        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5
        });
        
        setHandLandmarker(landmarker);
      } catch (err) {
        console.error('Error initializing HandLandmarker:', err);
        setError('Failed to initialize hand tracking.');
      }
    };

    initializeHandLandmarker();
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

  // Hand detection loop
  useEffect(() => {
    if (!handLandmarker || !videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let lastVideoTime = -1;

    const detectHands = async () => {
      if (!video || video.paused || video.ended) {
        animationFrameRef.current = requestAnimationFrame(detectHands);
        return;
      }

      // Set canvas size to match video
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;

      const currentTime = video.currentTime;
      if (currentTime !== lastVideoTime) {
        lastVideoTime = currentTime;

        // Detect hands
        const results = handLandmarker.detectForVideo(video, Date.now());

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw hand landmarks
        if (results.landmarks) {
          const drawingUtils = new DrawingUtils(ctx);

          for (const landmarks of results.landmarks) {
            drawingUtils.drawConnectors(
              landmarks,
              HandLandmarker.HAND_CONNECTIONS,
              { color: '#00FF00', lineWidth: 5 }
            );
            drawingUtils.drawLandmarks(landmarks, {
              color: '#FF0000',
              lineWidth: 2,
              radius: 5
            });
          }

          // Calculate distance if two hands are detected
          if (results.landmarks.length === 2) {
            const hand1 = results.landmarks[0];
            const hand2 = results.landmarks[1];

            // Use middle finger tip landmarks (index 12) to calculate distance
            const finger1 = hand1[12];
            const finger2 = hand2[12];

            // Calculate Euclidean distance
            const dx = (finger2.x - finger1.x) * canvas.width;
            const dy = (finger2.y - finger1.y) * canvas.height;
            const dz = (finger2.z - finger1.z) * canvas.width; // z is relative to wrist depth
            
            const distancePixels = Math.sqrt(dx * dx + dy * dy + dz * dz);
            
            // Convert to approximate real-world distance (rough estimation)
            // Assuming average hand width is about 85mm and video width represents ~500mm at arm's length
            const distanceCm = (distancePixels / canvas.width) * 50;
            
            setDistance(distanceCm);

            // Draw line between middle finger tips
            ctx.beginPath();
            ctx.moveTo(finger1.x * canvas.width, finger1.y * canvas.height);
            ctx.lineTo(finger2.x * canvas.width, finger2.y * canvas.height);
            ctx.strokeStyle = '#FFFF00';
            ctx.lineWidth = 3;
            ctx.stroke();
          } else {
            setDistance(null);
          }
        }
      }

      animationFrameRef.current = requestAnimationFrame(detectHands);
    };

    detectHands();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [handLandmarker]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center bg-gray-900 p-4">
      <div className="w-full max-w-5xl">
        <h1 className="text-4xl font-bold text-white text-center mb-8">
          MediaPipe Hand Distance Tracker
        </h1>
        
        <div className="relative bg-black rounded-lg overflow-hidden shadow-2xl">
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-gray-800 z-10">
              <div className="text-white text-xl">Loading camera and hand tracking model...</div>
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

        <div className="mt-6 p-6 bg-gray-800 rounded-lg">
          <div className="flex flex-col items-center space-y-4">
            <div className="text-center">
              <h2 className="text-xl font-semibold text-white mb-2">Distance Between Middle Fingers</h2>
              {distance !== null ? (
                <div className="text-5xl font-bold text-green-400">
                  {distance.toFixed(1)} cm
                </div>
              ) : (
                <div className="text-2xl text-gray-400">
                  Show both hands to measure distance
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
