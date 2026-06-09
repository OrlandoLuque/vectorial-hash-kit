use redis::{Client, Commands, Connection};
use std::sync::Mutex;

/// Redis-based task coordination and template storage for multi-process support
pub struct RedisStore {
    conn: Mutex<Connection>,
    pub lock_ttl: usize, // seconds
}

impl RedisStore {
    pub fn connect(host: &str, port: u16) -> Result<Self, String> {
        let url = format!("redis://{}:{}", host, port);
        let client = Client::open(url).map_err(|e| format!("Redis connect error: {}", e))?;
        let conn = client.get_connection().map_err(|e| format!("Redis connection error: {}", e))?;
        Ok(RedisStore {
            conn: Mutex::new(conn),
            lock_ttl: 50,
        })
    }

    /// Try to lock a task atomically. Returns true if we got the lock.
    pub fn try_lock_task(&self, task_key: &str) -> bool {
        let mut conn = self.conn.lock().unwrap();
        // SETNX pattern: only set if not exists
        let exists: bool = conn.exists(task_key).unwrap_or(true);
        if !exists {
            let _: () = conn.set_ex(task_key, "working", self.lock_ttl as u64).unwrap_or(());
            true
        } else {
            false
        }
    }

    /// Refresh lock TTL (keep-alive during processing)
    pub fn keep_lock(&self, task_key: &str) {
        let mut conn = self.conn.lock().unwrap();
        let _: () = conn.set_ex(task_key, "working", self.lock_ttl as u64).unwrap_or(());
    }

    /// Mark task as permanently completed
    pub fn complete_task(&self, task_key: &str) {
        let mut conn = self.conn.lock().unwrap();
        let _: () = conn.set(task_key, "completed").unwrap_or(());
    }

    /// Store a template with 8-symmetry deduplication (atomic via pipeline)
    /// Returns (template_id, is_new)
    pub fn store_template(&self, hashes: &[Vec<u8>], gen_record: &str,
                          count_key: &str, list_key: &str, gen_key: &str) -> (u32, bool) {
        let mut conn = self.conn.lock().unwrap();

        // Check if any of 8 hashes already exists
        let hash_strs: Vec<String> = hashes.iter().map(|h| hex::encode(h)).collect();

        for hash_str in &hash_strs {
            let result: Option<u32> = conn.get(hash_str).unwrap_or(None);
            if let Some(id) = result {
                // Duplicate found — record generation and return
                let _: () = conn.rpush(gen_key, gen_record).unwrap_or(());
                return (id, false);
            }
        }

        // New template — increment counter, store hash, record
        let id: u32 = conn.incr(count_key, 1).unwrap_or(1);
        let _: () = conn.set(&hash_strs[0], id).unwrap_or(());
        let _: () = conn.rpush(list_key, &hash_strs[0]).unwrap_or(());
        let _: () = conn.rpush(gen_key, gen_record).unwrap_or(());
        (id, true)
    }

    /// Save last processed position for resume
    pub fn save_progress(&self, key: &str, value: &str) {
        let mut conn = self.conn.lock().unwrap();
        let _: () = conn.set(key, value).unwrap_or(());
    }

    /// Get template count
    pub fn get_template_count(&self, key: &str) -> u32 {
        let mut conn = self.conn.lock().unwrap();
        conn.get(key).unwrap_or(0)
    }
}

// hex encoding for binary hashes
mod hex {
    pub fn encode(data: &[u8]) -> String {
        data.iter().map(|b| format!("{:02x}", b)).collect()
    }
}
