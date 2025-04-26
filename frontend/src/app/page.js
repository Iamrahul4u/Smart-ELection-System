// src/app/page.js
"use client";

import React, { useState, useEffect, useRef, useCallback } from 'react';
import io from 'socket.io-client';

// --- Configuration ---
const SERVER_URL = 'http://localhost:5000';
const FRAME_RATE = 10;
const PARTIES = ["BJP", "CONGRESS", "AAP", "NOTA"];

// --- Socket Instance ---
let socket = null;

function Home() {
    // --- State Variables ---
    const [isConnected, setIsConnected] = useState(false);
    const [serverMessage, setServerMessage] = useState('Connecting...');
    const [mode, setMode] = useState('idle');
    const [aadharInput, setAadharInput] = useState('');
    const [isCapturing, setIsCapturing] = useState(false);
    const [enrollStatus, setEnrollStatus] = useState({ message: '', success: null, readyToSave: false });
    const [enrollProgress, setEnrollProgress] = useState({ count: 0, total: 50, box: null });
    const [voteStatus, setVoteStatus] = useState({ status: 'idle', message: 'Initializing...', aadhar_display: null, can_vote: false, box: null });
    const [lastVoteResult, setLastVoteResult] = useState(null);
    const [trainingStatus, setTrainingStatus] = useState({ status: 'idle', message: '', running: false });
    const [isModelLoaded, setIsModelLoaded] = useState(false);

    // --- Refs ---
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const streamRef = useRef(null);
    const sendFrameIntervalRef = useRef(null);
    const isReadyToSaveRef = useRef(false); // Ref to track readiness immediately
    const isMounted = useRef(true);
    useEffect(() => { isMounted.current = true; return () => { isMounted.current = false; }; }, []);


    // --- Callbacks for Helpers ---
    const cleanupStream = useCallback(() => {
        console.log("Cleanup Stream Called");
        if (sendFrameIntervalRef.current) { clearInterval(sendFrameIntervalRef.current); sendFrameIntervalRef.current = null; console.log("Frame interval cleared."); }
        if (streamRef.current) { streamRef.current.getTracks().forEach(track => track.stop()); streamRef.current = null; console.log("Media stream tracks stopped."); }
        const video = videoRef.current; if (video && video.srcObject) { video.srcObject = null; console.log("Video srcObject cleared."); }
        setIsCapturing(false);
        const canvas = canvasRef.current; if (canvas) { try { const ctx = canvas.getContext('2d'); if (ctx) { ctx.clearRect(0, 0, canvas.width, canvas.height); console.log("Canvas cleared."); } } catch (e) { console.error("Error clearing canvas:", e); } }
    }, []);

    const startVideoStream = useCallback(async () => {
        console.log("Attempting start video stream...");
        if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) { console.error("getUserMedia not supported"); setServerMessage("Webcam access not supported or denied."); throw new Error("getUserMedia not supported"); }
        try {
            console.log("Requesting user media..."); const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } }); streamRef.current = stream; console.log("User media obtained.");
            if (!videoRef.current) { console.error("Video element Ref is NULL when trying to attach stream."); stream.getTracks().forEach(track => track.stop()); streamRef.current = null; throw new Error("Video element not found in DOM."); }
            console.log("Attaching stream to video element..."); videoRef.current.srcObject = stream;
            return new Promise((resolve, reject) => {
                const videoElement = videoRef.current; const timeoutDuration = 10000;
                const handleMetadataLoaded = () => { clearTimeout(timeoutId); if (!videoElement) { reject(new Error("Video element lost")); return; } console.log("Video metadata loaded. Attempting play..."); videoElement.play().then(() => { console.log("Video playback started."); setIsCapturing(true); const canvas = canvasRef.current; if (canvas && videoElement.videoWidth > 0) { canvas.width = videoElement.videoWidth; canvas.height = videoElement.videoHeight; } videoElement.onloadedmetadata = null; videoElement.onerror = null; resolve(); }).catch(err => { clearTimeout(timeoutId); console.error("Video play error:", err); setServerMessage(`Video play failed: ${err.message}`); cleanupStream(); videoElement.onloadedmetadata = null; videoElement.onerror = null; reject(err); }); };
                const handleError = (e) => { clearTimeout(timeoutId); console.error("Video element error:", e); setServerMessage("Error loading video stream."); cleanupStream(); if (videoElement) { videoElement.onloadedmetadata = null; videoElement.onerror = null; } reject(new Error("Video element error")); };
                const timeoutId = setTimeout(() => { console.warn(`Video metadata load timeout after ${timeoutDuration}ms.`); handleError(new Error("Metadata load timeout")); }, timeoutDuration);
                videoElement.onloadedmetadata = handleMetadataLoaded; videoElement.onerror = handleError;
            });
        } catch (error) { console.error("Error in startVideoStream:", error); let message = `Webcam Error: ${error.name}.`; if (error.message === "Video element not found in DOM.") { message = "Error initializing camera view." } else if (error.name === "NotAllowedError") { message = "Webcam permission denied."; } else if (error.name === "NotFoundError") { message = "No webcam found."; } setServerMessage(message); cleanupStream(); throw error; }
    }, [cleanupStream]);

    // --- Action Handlers ---
    const handleModeChange = useCallback((newMode) => {
        if (!isConnected && newMode !== 'idle') { setServerMessage("Not connected."); return; }
        console.log(`Button Click: Set Mode to ${newMode}`);
        isReadyToSaveRef.current = false; // Reset ready flag
    
        if (newMode === 'enroll_prompt') {
             setAadharInput('');
             // Optionally tell backend to go idle if needed
             // if (socket) socket.emit('set_mode', { mode: 'idle' });
        }
        else if (newMode === 'vote_recognize') {
           if (!isModelLoaded) {
               setServerMessage("Model not loaded.");
               return; // Don't change mode if model isn't ready
           }
           // **** TELL THE BACKEND TO CHANGE MODE ****
           if (socket) {
               console.log(">>> Emitting set_mode: vote_recognize to backend"); // Add log
               socket.emit('set_mode', { mode: 'vote_recognize' });
           }
           // **** END CHANGE ****
        } else if (newMode === 'idle') {
            // Tell backend to go idle too
            if (socket) {
               console.log(">>> Emitting set_mode: idle to backend"); // Add log
               socket.emit('set_mode', { mode: 'idle' });
           }
        }
    
        // Update local frontend state *after* potentially sending event
        setMode(newMode);
    
    }, [isConnected, isModelLoaded]); // Dependencies: isConnected, isModelLoaded (socket is stable)
    
    const handleStartEnrollment = useCallback(() => {
        if (!isConnected) { setServerMessage("Not connected."); return; }
        if (!aadharInput || aadharInput.length !== 12 || !/^\d+$/.test(aadharInput)) {
            setEnrollStatus({ message: 'Please enter a valid 12-digit Aadhar number.', success: false, readyToSave: false }); return;
        }
        console.log("Start Capture button clicked.");
        isReadyToSaveRef.current = false;
        setEnrollStatus({ message: 'Preparing camera...', success: null, readyToSave: false });
        setMode('enroll_capture');
    }, [isConnected, aadharInput]);

    // Use Ref check inside handler, no useCallback needed here
    const handleSaveEnrollment = () => {
        console.log(`handleSaveEnrollment CLICKED! Mode='${mode}', Ref says ReadyToSave=${isReadyToSaveRef.current}, isConnected=${isConnected}`);
        if (!isConnected) { setServerMessage("Not connected."); return; }
        if (mode === 'enroll_capture' && isReadyToSaveRef.current === true) { // Check the Ref
            console.log("Save condition PASSED (using Ref).");
            isReadyToSaveRef.current = false;
            setEnrollStatus(prev => ({...prev, message: 'Saving...', readyToSave: false}));
            if (socket) socket.emit('save_enrollment');
         } else {
             console.warn(`Save condition FAILED (using Ref). Mode check: ${mode === 'enroll_capture'}, Ready check: ${isReadyToSaveRef.current}`);
             setServerMessage("Cannot save yet.");
         }
    };

    const handleCancelEnrollment = useCallback(() => {
        console.log("Cancel Enrollment button clicked.");
        isReadyToSaveRef.current = false;
        setMode('idle');
    }, []);

    const handleCastVote = useCallback((party) => {
                if (!isConnected) { setServerMessage("Not connected."); return; }
                if (mode === 'vote_recognize' && voteStatus.can_vote && voteStatus.status === 'recognized') {
                    if (window.confirm(`Confirm vote for ${party}?`)) {
                        console.log(`Casting vote for ${party}`);
                        setVoteStatus(prev => ({...prev, can_vote: false, message: `Casting vote for ${party}...`}));
                        if (socket) socket.emit('cast_vote', { party: party });
                    }
                } else { console.warn("Cannot vote now."); }
            }, [isConnected, mode, voteStatus.can_vote, voteStatus.status]);
    const handleTrainModel = useCallback(() => {
                if (!isConnected) { setServerMessage("Not connected."); return; }
                if (!trainingStatus.running) {
                    if (window.confirm("Training requires enrolled data. Proceed?")) {
                        console.log("Train Model button clicked. Confirm accepted."); // Modified log
                        setTrainingStatus({ status: 'requested', message: 'Training requested...', running: true });
                        if (socket) {
                            socket.emit('train_model');
                            console.log("Frontend emitted 'train_model'"); // ADD THIS LOG
                        }
                    } else {
                        console.log("Train Model button clicked. Confirm rejected."); // Add log for rejected confirm
                    }
                } else { setServerMessage("Training is already in progress."); console.log("Train Model button clicked but training busy."); } // Modified log
            }, [isConnected, trainingStatus.running]);

    // --- Effects ---
    // Effect: Initialize Socket
    useEffect(() => {
        if (typeof window !== 'undefined' && !socket) {
            console.log("Effect: Initializing socket...");
            socket = io(SERVER_URL, { reconnectionAttempts: 5, transports: ['websocket'] });
            const handleDisconnect = (reason) => { if(isMounted.current) { setIsConnected(false); setServerMessage(`Disconnected: ${reason}`); setMode('idle'); cleanupStream(); console.log("Socket disconnected event", reason); }};
            socket.on('connect', () => { if(isMounted.current) { setIsConnected(true); setServerMessage('Connected.'); console.log("Socket connected event"); }});
            socket.on('disconnect', handleDisconnect);
            socket.on('connect_error', (error) => { if(isMounted.current) { setIsConnected(false); setServerMessage(`Connection Error: ${error.message}`); console.error("Socket Connect Error event:", error); }});
        }
        return () => {
            isReadyToSaveRef.current = false;
            if (socket && socket.connected) { console.log("Effect cleanup: Disconnecting socket..."); socket.disconnect(); }
            socket = null;
        };
    }, [cleanupStream]);

    // Effect: Setup Main Listeners
    useEffect(() => {
        if (!socket || !isConnected) return;
        console.log("Effect: Registering application listeners...");

        // **** MODIFIED LISTENER ****
        const onEnrollStatus = (data) => {
            console.log(`Event: Enroll Status Received (Raw):`, data);
            // Immediately capture the boolean value from the received data
            const isReadyNow = data?.ready_to_save === true; 
            console.log(`Local 'isReadyNow' captured as: ${isReadyNow}`);

            // Update Ref immediately using the captured value
            console.log(`Updating isReadyToSaveRef from ${isReadyToSaveRef.current} to ${isReadyNow}`);
            isReadyToSaveRef.current = isReadyNow;

            // Schedule state update using the captured value (for UI consistency)
            if (isMounted.current) {
                 console.log(`>>> Calling setEnrollStatus using local isReadyNow=${isReadyNow}. Data message=${data?.message}`);
                 // Use functional update form, but set readyToSave from the local variable
                 setEnrollStatus(prev => ({
                     ...prev, // Keep other existing state properties
                     message: data?.message ?? prev.message, // Update message from data
                     success: typeof data?.success === 'boolean' ? data.success : prev.success, // Update success from data
                     readyToSave: isReadyNow // Set readyToSave based on the captured local variable
                 }));
                 console.log(`>>> Called setEnrollStatus using local isReadyNow=${isReadyNow}.`);
            } else {
                 console.log("Component unmounted, skipping setEnrollStatus");
            }
        };
        // **** END MODIFIED LISTENER ****

        const onEnrollProgress = (data) => { if(isMounted.current) setEnrollProgress(prev => ({ ...prev, ...data })); };
        const onEnrollSaveStatus = (data) => { console.log("Event: Save Status:", data); if(isMounted.current){ setServerMessage(data.message); if(data.success){ isReadyToSaveRef.current = false; setMode('idle'); setAadharInput(''); cleanupStream();} } };
        const onFaceDetected = (data) => { if (mode === 'enroll_capture' && isMounted.current) { setEnrollProgress(prev => ({ ...prev, box: data.box })); }};
        const onVoteStatus = (data) => { if (mode === 'vote_recognize' && isMounted.current) { setVoteStatus(data); } };
        const onVoteResult = (data) => { console.log("Event: Vote Result:", data); if(isMounted.current){ setLastVoteResult(data); setServerMessage(data.message); if(data.success){ isReadyToSaveRef.current = false; setMode('idle'); cleanupStream(); setTimeout(() => setLastVoteResult(null), 5000); } else { setMode('idle'); cleanupStream(); } }};
        const onTrainingStatus = (data) => { console.log("Event: Train Status:", data); if(isMounted.current){ setTrainingStatus({status: data.status, message: data.message, running: data.status === 'started' || data.status === 'busy'}); setServerMessage(data.message); if(data.status === 'completed' && data.success){ setIsModelLoaded(true);} }};
        const onModelStatus = (data) => { console.log("Event: Model Status:", data); if(isMounted.current){ setIsModelLoaded(data.loaded);}};
        const onConnectionAck = (data) => { if(isMounted.current) setServerMessage(data.message);};
        const onError = (data) => { console.error("Server Error:", data.message); if(isMounted.current) setServerMessage(`Server Error: ${data.message}`); };
        const onModeSet = (data) => { console.log("Mode set by server:", data.mode); if(isMounted.current) setMode(data.mode); };

        socket.on('enroll_status', onEnrollStatus);
        socket.on('enroll_progress', onEnrollProgress);
        socket.on('enroll_save_status', onEnrollSaveStatus);
        socket.on('face_detected', onFaceDetected);
        socket.on('vote_status', onVoteStatus);
        socket.on('vote_result', onVoteResult);
        socket.on('training_status', onTrainingStatus);
        socket.on('model_status', onModelStatus);
        socket.on('connection_ack', onConnectionAck);
        socket.on('error', onError);
        socket.on('mode_set', onModeSet);

         return () => { // Cleanup listeners
             if (!socket) return;
             console.log("Effect cleanup: Removing application listeners...");
             socket.off('enroll_status'); socket.off('enroll_progress'); socket.off('enroll_save_status');
             socket.off('face_detected'); socket.off('vote_status'); socket.off('vote_result');
             socket.off('training_status'); socket.off('model_status'); socket.off('connection_ack');
             socket.off('error'); socket.off('mode_set');
         };
    }, [isConnected, mode, cleanupStream]); // Dependencies

    // Effect: Manage Video Stream
    useEffect(() => {
        if (mode === 'enroll_capture' || mode === 'vote_recognize') {
            console.log(`Effect: Mode is ${mode}. Starting video...`);
            isReadyToSaveRef.current = false;
            cleanupStream();
            let didCancel = false;
            startVideoStream()
                .then(() => {
                    if (didCancel || !isMounted.current) return;
                    console.log(`Effect: Video started for mode: ${mode}`);
                    if (mode === 'enroll_capture' && socket && socket.connected && aadharInput) {
                        console.log("Effect: Emitting start_enroll");
                        socket.emit('start_enroll', { aadhar: aadharInput });
                        setEnrollStatus(prev => ({...prev, message: 'Camera active.', success: true, readyToSave: false }));
                    } else if (mode === 'vote_recognize') {
                         setVoteStatus(prev => ({ ...prev, message: 'Place face in camera.', status: 'no_face', can_vote: false }));
                    }
                })
                .catch(err => {
                    if (didCancel || !isMounted.current) return;
                    console.error(`Effect: Failed start video [${mode}]:`, err);
                    setMode(prevMode => prevMode === 'enroll_capture' ? 'enroll_prompt' : 'idle');
                });
             return () => { didCancel = true; console.log(`Effect cleanup: Mode changing from ${mode}, cleaning stream.`); cleanupStream(); };
        }
    }, [mode, aadharInput, startVideoStream, cleanupStream]);

    // Effect: Send Video Frames
    useEffect(() => {
        if (!isCapturing || (mode !== 'enroll_capture' && mode !== 'vote_recognize')) {
            if (sendFrameIntervalRef.current) { console.log("Effect condition false: Clearing frame sending interval."); clearInterval(sendFrameIntervalRef.current); sendFrameIntervalRef.current = null; }
            return;
        }
        console.log("Effect: Starting frame sending interval.");
        const videoElement = videoRef.current; const canvasElement = canvasRef.current;
        if (!videoElement || !canvasElement) { console.warn("Video/Canvas ref not ready."); return; }
        if (canvasElement.width === 0 || canvasElement.height === 0) { if (videoElement.videoWidth > 0) { canvasElement.width = videoElement.videoWidth; canvasElement.height = videoElement.videoHeight; } else { console.warn("Canvas/Video dimensions not ready."); return; } }
        let context; try { context = canvasElement.getContext('2d'); } catch(e) { console.error("Canvas context error:", e); return; } if (!context) { console.error("Canvas context is null."); return; }
        if (sendFrameIntervalRef.current) { clearInterval(sendFrameIntervalRef.current); }

        const intervalId = setInterval(() => {
            if (!isCapturing || !isMounted.current || !videoElement || !canvasElement || !context || videoElement.paused || videoElement.ended || videoElement.readyState < 3) { return; }
            try {
                context.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);
                let box = null; if (mode === 'vote_recognize' && voteStatus.box) box = voteStatus.box; else if (mode === 'enroll_capture' && enrollProgress.box) box = enrollProgress.box;
                if (box) { const { x, y, w, h } = box; let color = 'lime'; if (mode === 'vote_recognize') { if (voteStatus.status === 'recognized') color = 'lime'; else if (voteStatus.status === 'already_voted') color = 'red'; else color = 'yellow'; } context.strokeStyle = color; context.lineWidth = 3; context.strokeRect(x, y, w, h); if (mode === 'vote_recognize' && voteStatus.aadhar_display) { context.fillStyle = color; context.font = '16px Arial'; context.fillText(voteStatus.aadhar_display, x, y > 20 ? y - 5 : y + h + 15); } }
                const imageDataURL = canvasElement.toDataURL('image/jpeg', 0.7);
                if (socket && socket.connected) { socket.emit('video_frame', { image_data_url: imageDataURL }); }
                else { console.warn("Socket disconnected during frame sending."); }
            } catch (drawError) { console.error('Error during frame draw/send:', drawError); }
        }, 1000 / FRAME_RATE);
        sendFrameIntervalRef.current = intervalId;
        return () => { if (sendFrameIntervalRef.current) { console.log("Effect cleanup: Clearing frame sending interval."); clearInterval(sendFrameIntervalRef.current); sendFrameIntervalRef.current = null; }};
    }, [isCapturing, mode, voteStatus, enrollProgress]);


    // --- Render Logic ---
    // Button appearance is driven by state
    const isSaveDisabled = !enrollStatus.readyToSave;
    // Log state/ref values during render
    // console.log(`--- Rendering Home --- Mode: ${mode}, State readyToSave: ${enrollStatus.readyToSave}, Ref readyToSave: ${isReadyToSaveRef.current}, Calculated isSaveDisabled: ${isSaveDisabled}`);

    return (
        <div className="app-container" style={{ fontFamily: 'sans-serif', textAlign: 'center', paddingBottom: '30px' }}>
             {/* Header */}
             <header style={{ backgroundColor: '#282c34', padding: '15px', color: 'white', marginBottom: '20px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                 <h1>Facial Recognition Voting System</h1>
                 <div style={{ padding: '5px 10px', borderRadius: '5px', fontWeight: 'bold', backgroundColor: isConnected ? 'lightgreen' : 'lightcoral', color: isConnected ? 'darkgreen' : 'darkred' }}>
                     {isConnected ? 'Connected' : 'Disconnected'}
                 </div>
             </header>

             <main style={{ padding: '0 20px' }}>
                 {/* Controls */}
                 <div className="controls" style={{ marginBottom: '20px' }}>
                     <button style={{ margin: '5px 10px', padding: '10px 15px', cursor: 'pointer' }} onClick={() => handleModeChange('enroll_prompt')} disabled={mode === 'enroll_prompt' || mode === 'enroll_capture' || trainingStatus.running}>Enroll New Voter</button>
                     <button style={{ margin: '5px 10px', padding: '10px 15px', cursor: 'pointer' }} onClick={() => handleModeChange('vote_recognize')} disabled={!isModelLoaded || mode === 'vote_recognize' || trainingStatus.running}>Start Voting Session</button>
                     <button style={{ margin: '5px 10px', padding: '10px 15px', cursor: 'pointer' }} onClick={handleTrainModel} disabled={trainingStatus.running}>{trainingStatus.running ? 'Training...' : 'Train Recognition Model'}</button>
                     {mode !== 'idle' && (<button style={{ margin: '5px 10px', padding: '10px 15px', cursor: 'pointer' }} onClick={() => handleModeChange('idle')} disabled={trainingStatus.running}>Back to Idle</button>)}
                 </div>

                 {/* Status Message Area */}
                 <div className="status-message" style={{ margin: '15px 0', padding: '10px', backgroundColor: '#f0f0f0', borderLeft: '5px solid #ccc', textAlign: 'left', minHeight: '40px' }}>
                     <p style={{ margin: '0 0 5px 0' }}><strong>Status:</strong> {serverMessage || 'Welcome!'}</p>
                     {trainingStatus.message && <p style={{ margin: '0 0 5px 0', fontStyle: 'italic' }}>Training: {trainingStatus.message}</p>}
                     {(mode === 'enroll_prompt' || mode === 'enroll_capture') && enrollStatus.message && enrollStatus.message !== 'Camera active.' && <p style={{ margin: 0, color: enrollStatus.success === false ? 'red' : 'inherit' }}>{enrollStatus.message}</p>}
                     {mode === 'vote_recognize' && voteStatus.message && <p style={{ margin: 0 }}>{voteStatus.message}</p>}
                     {lastVoteResult && <p style={{ margin: 0, color: lastVoteResult.success ? 'green' : 'red' }}>Last Vote: {lastVoteResult.message}</p>}
                 </div>

                 {/* Content Area */}
                 <div className="content-area" style={{ marginTop: '20px', padding: '20px', border: '1px solid #ddd', minHeight: '500px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>

                     {/* Enrollment Prompt UI */}
                     {mode === 'enroll_prompt' && (
                         <div className="enroll-prompt">
                             <h2>Enter Aadhar to Enroll</h2>
                             <input type="text" value={aadharInput} onChange={(e) => setAadharInput(e.target.value)} placeholder="12-digit Aadhar Number" maxLength="12" style={{ padding: '8px', margin: '10px', width: '250px', fontSize:'1em' }} />
                             <button onClick={handleStartEnrollment} style={{ padding: '8px 15px', fontSize:'1em', cursor: 'pointer' }}>Start Capture</button>
                         </div>
                     )}

                     {/* Enrollment Capture UI */}
                     {mode === 'enroll_capture' && (
                         <div className="enroll-capture">
                             <h2>Capturing Face for Aadhar ...{aadharInput.slice(-4)}</h2>
                             <div className="video-container" style={{ position: 'relative', width: '640px', height: '480px', margin: '15px auto', backgroundColor: '#333', border: '1px solid #ccc', overflow: 'hidden' }}>
                                 <video ref={videoRef} playsInline muted autoPlay style={{ display: 'block', width: '100%', height: '100%', objectFit: 'cover' }}></video>
                                 <canvas ref={canvasRef} className="video-overlay" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}></canvas>
                                 {!isCapturing && <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'rgba(0,0,0,0.7)', display: 'flex', justifyContent: 'center', alignItems: 'center', color: 'white', fontSize: '1.2em' }}><p>Starting camera...</p></div>}
                             </div>
                             <progress value={enrollProgress.count} max={enrollProgress.total} style={{ width: '80%', maxWidth: '400px', margin: '15px 0 5px 0' }}></progress>
                             <p style={{margin: '0 0 15px 0'}}>{enrollProgress.count} / {enrollProgress.total} frames captured</p>

                             {/* Buttons */}
                             <button
                                 onClick={handleSaveEnrollment} // Click handler checks Ref internally
                                 disabled={isSaveDisabled} // Appearance based on State
                                 style={{
                                     margin: '5px', padding: '10px 15px', fontSize:'1em',
                                     cursor: !isSaveDisabled ? 'pointer' : 'default',
                                     opacity: !isSaveDisabled ? 1 : 0.6,
                                     border: !isSaveDisabled ? '2px solid green' : '1px solid gray'
                                 }}
                             >
                                 Save Enrollment
                             </button>
                             <button onClick={handleCancelEnrollment} style={{ margin: '5px', padding: '10px 15px', fontSize:'1em', cursor: 'pointer' }}>Cancel</button>
                         </div>
                     )}

                    {/* Voting Recognition UI */}
                     {mode === 'vote_recognize' && (
                         <div className="vote-recognize">
                             <h2>Voting Session</h2>
                              <div className="video-container" style={{ position: 'relative', width: '640px', height: '480px', margin: '15px auto', backgroundColor: '#333', border: '1px solid #ccc', overflow: 'hidden' }}>
                                 <video ref={videoRef} playsInline muted autoPlay style={{ display: 'block', width: '100%', height: '100%', objectFit: 'cover' }}></video>
                                 <canvas ref={canvasRef} className="video-overlay" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}></canvas>
                                 {!isCapturing && <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', backgroundColor: 'rgba(0,0,0,0.7)', display: 'flex', justifyContent: 'center', alignItems: 'center', color: 'white', fontSize: '1.2em' }}><p>Starting camera...</p></div>}
                             </div>
                             <div className="vote-buttons" style={{ marginTop: '20px' }}>
                                 {PARTIES.map(party => (
                                     <button key={party} onClick={() => handleCastVote(party)} disabled={!voteStatus.can_vote} style={{ margin: '5px', padding: '12px 20px', fontSize: '1.1em', cursor: voteStatus.can_vote ? 'pointer' : 'default' }}>
                                         Vote {party}
                                     </button>
                                 ))}
                             </div>
                         </div>
                     )}

                     {/* Idle Mode UI */}
                     {mode === 'idle' && !trainingStatus.running && (
                         <div className="idle-mode" style={{ paddingTop: '50px' }}>
                             <h2>Select an option above</h2>
                              {!isModelLoaded && <p style={{color: 'orange', fontWeight:'bold'}}>Warning: Recognition model not loaded. Enroll voters and then Train Model.</p>}
                         </div>
                     )}

                 </div> {/* End content-area */}
             </main>
        </div> // End app-container
    );
}

export default Home;