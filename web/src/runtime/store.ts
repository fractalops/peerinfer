
/** Content-addressed, cross-visit model cache. Identity and invitations do not affect cache keys. */
export class ModelStore {
  constructor(private db: IDBDatabase) {}
  static async open(): Promise<ModelStore> {
    return new Promise((resolve, reject) => {
      const r = indexedDB.open("peerinfer.models.v1", 1);
      r.onupgradeneeded = () => r.result.createObjectStore("objects");
      r.onerror = () =>
        reject(
          new Error(
            "Browser storage is unavailable. Allow site storage and try again.",
          ),
        );
      r.onsuccess = () => resolve(new ModelStore(r.result));
    });
  }
  read(key: string): Promise<Uint8Array | undefined> {
    return new Promise((resolve, reject) => {
      const r = this.db.transaction("objects").objectStore("objects").get(key);
      r.onsuccess = () => resolve(r.result);
      r.onerror = () => reject(r.error);
    });
  }
  write(key: string, bytes: Uint8Array): Promise<void> {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction("objects", "readwrite");
      tx.objectStore("objects").put(bytes.slice(), key);
      tx.oncomplete = () => resolve();
      tx.onerror = tx.onabort = () =>
        reject(
          new Error(
            "Could not save model weights. Free some browser storage and try again.",
          ),
        );
    });
  }
  close() {
    this.db.close();
  }
}
