import React, { useEffect, useState } from 'react';
import axios from 'axios';

const App = () => {
    const [documents, setDocuments] = useState([]);
    const [question, setQuestion] = useState('');
    const [role, setRole] = useState('');
    const [answer, setAnswer] = useState('');

    useEffect(() => {
        fetchDocuments();
    }, []);

    const fetchDocuments = async () => {
        const response = await axios.get('http://localhost:5000/documents');
        setDocuments(response.data.documents);
    };

    const handleDelete = async (docName) => {
        await axios.delete('http://localhost:5000/delete', { data: { doc_name: docName } });
        fetchDocuments();
    };

    const handleAsk = async () => {
        const response = await axios.post('http://localhost:5000/ask', { query: question, role });
        setAnswer(response.data.answer);
    };

    return (
        <div>
            <h1>PDF Question Answering App</h1>
            <div>
                <h2>Uploaded Documents</h2>
                {documents.map(doc => (
                    <div key={doc.doc_name}>
                        <p>{doc.doc_name}</p>
                        <button onClick={() => handleDelete(doc.doc_name)}>Delete</button>
                    </div>
                ))}
            </div>
            <div>
                <h2>Ask a Question</h2>
                <input type="text" value={question} onChange={e => setQuestion(e.target.value)} placeholder="Enter your question" />
                <input type="text" value={role} onChange={e => setRole(e.target.value)} placeholder="Enter the role" />
                <button onClick={handleAsk}>Ask</button>
                <p>{answer}</p>
            </div>
        </div>
    );
};

export default App;
