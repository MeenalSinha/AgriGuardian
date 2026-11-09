# AGRIGUARDIAN — PHASE 10 OFFLINE OPERATIONS GUIDE
## Field Officer & Farmer Field Manual for Low-Connectivity Operations

**Audience:** Agricultural Extension Officers, Krishi Vigyan Kendra (KVK) Staff, Village Resource Persons, Lead Farmers  
**Target App:** AgriGuardian Progressive Web App (PWA) at `/pwa` and Desktop Hybrid Portal  
**Document Revision:** 1.10.0 (November 2026)

---

## 1. Introduction

In rural agricultural clusters across India, internet connectivity is frequently unavailable or erratic. Whether standing in remote crop fields, visiting farmers in shadow zones, or operating in deep mandi yards, agricultural field officers and farmers must be able to:
- Access previously loaded farm and parcel profiles.
- Inspect crop growth stages and diagnostic histories.
- Log newly completed agronomic tasks (sowing, irrigation, fertilizer application).
- Mark scheduled smart farm reminders as completed or snoozed.
- Capture crop leaf disease photos with device cameras and queue them for analysis.
- Automatically synchronize data when reconnecting to Wi-Fi or 4G networks.

This manual provides practical instructions for installing, configuring, operating, and troubleshooting the AgriGuardian offline PWA.

---

## 2. Installing the Progressive Web App (PWA)

The AgriGuardian PWA requires **no app store download**. It installs directly from the browser onto Android, iOS, and desktop operating systems.

### 2.1 Android Devices (Google Chrome)
1. Open Google Chrome on your smartphone.
2. Navigate to your AgriGuardian URL and append `/pwa` (e.g. `http://<server-ip>:8000/pwa`).
3. Tap the **"📲 Install App"** button at the top of the screen, or tap the three dots (`⋮`) in the top right corner and choose **"Add to Home screen"** / **"Install app"**.
4. Confirm installation. The AgriGuardian green leaf icon will appear on your phone's home screen.
5. Tap the home screen icon to launch the app in standalone fullscreen mode.

### 2.2 Apple iOS Devices (Safari)
1. Open Safari and navigate to `/pwa`.
2. Tap the **Share** button (box with an upward arrow `↑`) at the bottom of the screen.
3. Scroll down and tap **"Add to Home Screen"**.
4. Name the application `AgriGuardian` and tap **Add**.
5. Launch from your home screen for an app-like fullscreen experience.

### 2.3 Desktop / Laptop (Chrome / Edge)
1. Navigate to `/pwa`.
2. Click the install icon (computer with down arrow) in the browser address bar.
3. Click **"Install"**. The app opens in an isolated desktop window.

---

## 3. Pre-Field Preparation (Before Leaving Strong Network)

To ensure smooth operation in the field, always perform the following **2-minute pre-field checklist** while connected to reliable office or home Wi-Fi:

1. **Log In:** Open the PWA and sign in with your credentials.
2. **Initial Bootstrap Sync:**
   - Tap the **"🔄 Sync"** tab in the bottom navigation bar.
   - Tap **"🔄 Sync Now"**.
   - Verify that your farms, crop cycles, scheduled activities, and pending reminders load into the local database.
3. **Verify Service Worker Caching:**
   - Turn on Airplane Mode on your phone to test.
   - Close and re-open the PWA.
   - Verify that the app shell loads immediately and the top status banner displays:  
     `🟠 Offline Mode — Last updated: Today, HH:MM (Offline)`
   - Turn Airplane Mode back off.

---

## 4. Operating in the Field (Offline Mode)

When working in zero-connectivity fields, AgriGuardian operates completely offline using client-side **IndexedDB**:

### 4.1 Viewing Farm Context & Crop Cycles
- Navigate to the **"📊 Dashboard"** tab.
- All registered farms, active crop cycles, and current vegetative/reproductive stages remain visible.
- Advisories and historical diagnoses are accessible without network requests.
- All cached data displays honest timestamps indicating when it was last fetched from the server.

### 4.2 Logging Farm Activities Offline
1. Tap the **"📝 Activities"** tab.
2. Select your Farm and Parcel/Plot.
3. Choose the Activity Type (e.g., *Irrigation, Pesticide Spray, Weeding, Fertilizer Application*).
4. Enter the date, cost incurred (₹), and any field notes.
5. Tap **"Save Activity (Offline)"**.
6. **Result:** The activity is saved immediately to IndexedDB and enqueued in the **Offline Mutation Queue**. You will see the queued mutation count badge increase (e.g. `🔄 (1)`).

### 4.3 Managing Smart Reminders Offline
1. Tap the **"⏰ Reminders"** tab.
2. Review tasks scheduled for today or upcoming days.
3. If completed in the field, tap **"✓ Complete"**.
4. If delayed due to local microclimate (e.g. high winds preventing pesticide spraying), tap **"⏳ Snooze"**.
5. Changes are persisted locally and queued for synchronization.

### 4.4 Capturing Crop Disease Photographs Offline
1. Tap the **"📷 Camera"** tab.
2. Tap **"Open Camera"** to activate your device's camera stream, or tap **"Choose from Gallery"**.
3. Frame the symptomatic leaf clearly in good lighting and tap **"Capture Photo"**.
4. The photo is automatically:
   - Resized and compressed to prevent local storage exhaustion.
   - Assigned a cryptographically secure SHA-256 integrity hash.
   - Enqueued with an offline diagnosis draft in the local queue.
5. Tap **"Queue Diagnosis Draft"**. The photo is securely retained in IndexedDB until connectivity returns.

---

## 5. Returning from the Field & Synchronizing Data

When you return to an area with 4G/5G or Wi-Fi coverage:

1. **Automatic Network Detection:**
   - AgriGuardian's background network monitor constantly pings the server.
   - As soon as connectivity is restored, the top banner updates to `🟢 Online`.
   - The app automatically triggers a background synchronization cycle.
2. **Manual Sync Trigger:**
   - You can manually initiate a sync at any time by tapping **"🔄 Sync"** -> **"🔄 Sync Now"**.
3. **Synchronization Phases:**
   - **Step 1 (Push):** All queued offline activities, reminder updates, and new records are transmitted to the server.
   - **Step 2 (Attachments):** Queued crop disease photographs are uploaded and linked to their diagnostic cases.
   - **Step 3 (Pull):** Incremental changes made on the server (e.g., agronomist reviews, updated weather forecasts) are pulled down to update local IndexedDB.
4. **Zero Data Loss Guarantee:** If the network disconnects halfway through a sync, completed mutations remain confirmed on the server, while unconfirmed mutations remain safely in the local queue for the next retry.

---

## 6. Resolving Data Conflicts

If another user (such as a senior agronomist at headquarters) modified the same reminder or activity while you were editing it offline, a **version collision** is detected:

1. The PWA top bar will display a notification: `⚠️ 1 Sync Conflict Pending Resolution`.
2. Tap the **"⚠️ Conflicts"** tab.
3. A side-by-side comparison screen will display:
   - **Your Offline Version (Client):** Date, status, and notes you entered in the field.
   - **Headquarters Version (Server):** Date, status, and notes updated by the agronomist.
4. **Choose Resolution Strategy:**
   - **"Accept My Version (Client Wins)":** Overwrites server with your field observations.
   - **"Accept Server Version (Server Wins)":** Keeps headquarters' updates.
   - **"Manual Merge":** Opens a form to combine field notes with agronomist instructions.
5. Tap **"Resolve Conflict"**. The resolved record is synchronized instantly.

---

## 7. Security, Tenant Isolation & Device Hygiene

- **Device Authorization:** Every mobile device must be registered. If a device is lost or stolen, administrators can instantly revoke it from the desktop portal at `/api/v1/sync/devices/{id}` or via Streamlit **Offline & Sync Center**.
- **Tenant Isolation:** Client IndexedDB and backend change logs enforce strict farmer-level scoping. A field officer logged in as Farmer A cannot view or pull Farmer B's records.
- **Shared Tablet Operations:** If multiple field officers share a single tablet:
  - Always tap **"Log Out"** before handing the tablet to another officer.
  - Logging out purges sensitive session tokens while preserving cached offline assets.
