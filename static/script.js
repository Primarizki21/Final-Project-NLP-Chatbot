
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

const userInput = document.getElementById('user-input');
const chatHistory = document.getElementById('chat-history');
const sendBtn = document.getElementById('send-btn');
const welcomeHero = document.getElementById('welcome-hero');
const newChatBtn = document.querySelector('.nav-btn.primary');
const chatListItems = document.getElementById('chat-list-items');
const sidebarToggle = document.getElementById('sidebar-toggle');
const renameModal = document.getElementById('rename-modal');
const renameInput = document.getElementById('rename-input');
const renameSaveBtn = document.getElementById('rename-save');
const renameCancelBtn = document.getElementById('rename-cancel');

let chats = [];
let activeChatId = null;
let renameTargetId = null;

function loadChats() {
    try {
        const saved = localStorage.getItem('tsd_chats');
        if (saved) {
            chats = JSON.parse(saved);
        }
    } catch (e) {
        console.error('Failed to load chats from localStorage', e);
        chats = [];
    }
    if (!Array.isArray(chats) || chats.length === 0) {
        const firstChat = {
            id: Date.now(),
            title: 'Chat 1 percobaan',
            messages: [],
        };
        chats = [firstChat];
        activeChatId = firstChat.id;
        saveChats();
    } else if (!activeChatId) {
        activeChatId = chats[0].id;
    }
}

function saveChats() {
    localStorage.setItem('tsd_chats', JSON.stringify(chats));
}

function renderChatList() {
    if (!chatListItems) return;
    chatListItems.innerHTML = '';
    chats.forEach((chat, index) => {
        const btn = document.createElement('button');
        btn.className = 'chat-item' + (chat.id === activeChatId ? ' active' : '');
        const title = chat.title || `Chat ${index + 1}`;
        btn.innerHTML = `
            <span class="chat-title">${title}</span>
            <span class="chat-rename-icon">✏️</span>
            <span class="chat-delete-icon">🗑</span>
        `;
        btn.onclick = () => switchChat(chat.id);
        const icon = btn.querySelector('.chat-rename-icon');
        if (icon) {
            icon.addEventListener('click', (e) => {
                e.stopPropagation();
                openRenameModal(chat.id);
            });
        }
        const delIcon = btn.querySelector('.chat-delete-icon');
        if (delIcon) {
            delIcon.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteChat(chat.id);
            });
        }
        chatListItems.appendChild(btn);
    });
}

function renderActiveChat() {
    const current = chats.find(c => c.id === activeChatId);
    if (!current) return;

    chatHistory.innerHTML = '';
    if (current.messages.length === 0 && welcomeHero) {
        welcomeHero.style.display = 'block';
    } else if (welcomeHero) {
        welcomeHero.style.display = 'none';
    }

    current.messages.forEach(msg => {
        appendMessage(msg.role, msg.text, false);
    });
}

function openRenameModal(id) {
    const chat = chats.find(c => c.id === id);
    if (!chat) return;
    renameTargetId = id;
    if (renameInput) {
        renameInput.value = chat.title || '';
        renameInput.focus();
        renameInput.select();
    }
    if (renameModal) {
        renameModal.style.display = 'flex';
    }
}

function closeRenameModal() {
    renameTargetId = null;
    if (renameModal) {
        renameModal.style.display = 'none';
    }
}

function applyRename() {
    if (!renameTargetId || !renameInput) return;
    const newTitle = renameInput.value.trim();
    if (!newTitle) {
        closeRenameModal();
        return;
    }
    const chat = chats.find(c => c.id === renameTargetId);
    if (!chat) {
        closeRenameModal();
        return;
    }
    chat.title = newTitle;
    chat._hasTitleFromUser = true;
    saveChats();
    renderChatList();
    closeRenameModal();
}

function deleteChat(id) {
    if (chats.length <= 1) {
        alert('Minimal harus ada satu chat.');
        return;
    }
    const chat = chats.find(c => c.id === id);
    const name = chat?.title || 'chat ini';
    if (!window.confirm(`Hapus ${name}?`)) {
        return;
    }
    chats = chats.filter(c => c.id !== id);
    if (activeChatId === id) {
        activeChatId = chats[0]?.id || null;
    }
    saveChats();
    renderChatList();
    renderActiveChat();
}

function switchChat(id) {
    activeChatId = id;
    renderChatList();
    renderActiveChat();
}

function createNewChat() {
    const idx = chats.length + 1;
    const chat = {
        id: Date.now(),
        title: `Chat ${idx} percobaan`,
        messages: [],
    };
    chats.unshift(chat);
    activeChatId = chat.id;
    saveChats();
    renderChatList();
    renderActiveChat();
}

// Send on Enter (Shift+Enter for newline)
userInput.addEventListener('keydown', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

function formatMarkdown(text) {
    if (!text) return '';
    
    let html = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    html = html.replace(/(?:^\s*\*\s+[^\n]*(?:\n|$))+/gm, (match) => {
        const cleanMatch = match.trim();
        const items = cleanMatch.split('\n');
        
        const listItems = items.map(item => {
            return '<li>' + item.replace(/^\s*\*\s+/, '').trim() + '</li>';
        }).join('');
        
        return '<ul>' + listItems + '</ul>';
    });

    const placeholderPrefix = 'ⓅⓁⓐⓒⓔⓗⓞⓛⓓⓔⓡ';
    const placeholders = [];
    let placeholderIndex = 0;
    
    // Proses Bold (**)
    html = html.replace(/\*\*([^*\n]+?)\*\*/g, (match, content) => {
        if (!content.trim()) return match;
        const placeholder = placeholderPrefix + placeholderIndex++ + 'ⓔⓝⓓ';
        placeholders.push({ placeholder, replacement: '<strong>' + content + '</strong>' });
        return placeholder;
    });

    // Proses Bold dengan Underscore (__)
    html = html.replace(/__([^_\n]+?)__(?!_)/g, (match, content) => {
        if (!content.trim()) return match;
        if (/^MD\d+$/.test(content.trim())) {
            return match;
        }
        const placeholder = placeholderPrefix + placeholderIndex++ + 'ⓔⓝⓓ';
        placeholders.push({ placeholder, replacement: '<strong>' + content + '</strong>' });
        return placeholder;
    });
    
    html = html.replace(/\*([^*\n]+?)\*/g, (match, content) => {
        if (!content.trim()) return match;
        return '<em>' + content + '</em>';
    });
    
    html = html.replace(/(?<!_)_([^_\n]+?)_(?!_)/g, (match, content) => {
        if (!content.trim()) return match;
        return '<em>' + content + '</em>';
    });
    
    placeholders.forEach(({ placeholder, replacement }) => {
        html = html.split(placeholder).join(replacement);
    });
    
    const remainingPlaceholderPattern = new RegExp(placeholderPrefix.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\d+ⓔⓝⓓ', 'g');
    html = html.replace(remainingPlaceholderPattern, (match) => {
        console.warn('Unrestored placeholder found:', match);
        return '';
    });
    
    html = html.replace(/__MD\d+___/g, (match) => {
        console.warn('Model placeholder found (should have been replaced by model):', match);
        return '[Nama Mata Kuliah]';
    });
    
    // Ubah newline sisa menjadi <br>
    html = html.replace(/\n/g, '<br>');
    
    return html;
}

function appendMessage(role, text, save = true) {
    const msgDiv = document.createElement('div');
    msgDiv.className = `message ${role}-message`;
    
    const avatar = role === 'user' ? '👤' : '✨';
    
    const formattedText = formatMarkdown(text);

    msgDiv.innerHTML = `
        <div class="avatar">${avatar}</div>
        <div class="content"><p>${formattedText}</p></div>
    `;
    
    chatHistory.appendChild(msgDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    if (save) {
        const current = chats.find(c => c.id === activeChatId);
        if (current) {
            current.messages.push({ role, text });
            if (role === 'user' && !current._hasTitleFromUser) {
                current.title = text.length > 25 ? text.slice(0, 25) + '…' : text;
                current._hasTitleFromUser = true;
                renderChatList();
            }
            saveChats();
        }
    }
}

function showLoading() {
    const loadingDiv = document.createElement('div');
    loadingDiv.id = 'loading-indicator';
    loadingDiv.className = 'message system-message';
    loadingDiv.innerHTML = `
        <div class="avatar">✨</div>
        <div class="content">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>
    `;
    chatHistory.appendChild(loadingDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function removeLoading() {
    const loadingDiv = document.getElementById('loading-indicator');
    if (loadingDiv) loadingDiv.remove();
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    if (welcomeHero) {
        welcomeHero.style.display = 'none';
    }

    userInput.value = '';
    userInput.style.height = 'auto';
    userInput.disabled = true;
    sendBtn.disabled = true;

    appendMessage('user', text);
    showLoading();

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message: text })
        });

        const data = await response.json();
        removeLoading();
        appendMessage('system', data.response);

    } catch (error) {
        removeLoading();
        appendMessage('system', "Maaf, terjadi kesalahan koneksi.");
        console.error(error);
    } finally {
        userInput.disabled = false;
        sendBtn.disabled = false;
        userInput.focus();
    }
}

if (newChatBtn) {
    newChatBtn.addEventListener('click', () => {
        createNewChat();
    });
}

if (sidebarToggle) {
    sidebarToggle.addEventListener('click', () => {
        document.body.classList.toggle('sidebar-collapsed');
    });
}

if (renameCancelBtn) {
    renameCancelBtn.addEventListener('click', () => {
        closeRenameModal();
    });
}

if (renameSaveBtn) {
    renameSaveBtn.addEventListener('click', () => {
        applyRename();
    });
}

if (renameInput) {
    renameInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            e.preventDefault();
            applyRename();
        } else if (e.key === 'Escape') {
            e.preventDefault();
            closeRenameModal();
        }
    });
}

loadChats();
renderChatList();
renderActiveChat();

