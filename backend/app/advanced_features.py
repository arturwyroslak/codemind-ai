#!/usr/bin/env python3
"""Multimodal RAG Pipeline with Weaviate integration"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import io
import base64

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, HumanMessage
from weaviate import Client
from weaviate.classes.config import Configure, Property, DataType
from app.core.config import settings

logger = logging.getLogger(__name__)

class MultimodalRAG:
    """Advanced RAG pipeline supporting text, images, and multimodal queries"""
    
    def __init__(self):
        # Initialize clients
        self.llm = ChatOpenAI(
            model=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY,
            temperature=0.2
        )
        
        self.text_embedder = OpenAIEmbeddings(
            model="text-embedding-3-large",
            openai_api_key=settings.OPENAI_API_KEY
        )
        
        # Initialize Weaviate with multimodal support
        self.weaviate_client = self._init_weaviate()
        
        # Supported content types
        self.supported_types = {
            'text': self._process_text,
            'image': self._process_image,
            'diagram': self._process_diagram,
            'audio': self._process_audio,
            'video': self._process_video
        }
    
    def _init_weaviate(self) -> Client:
        """Initialize Weaviate client with multimodal schema"""
        client = Client(
            url="http://localhost:8080",
            auth_client_secret=None,  # Configure auth as needed
        )
        
        # Create schema if not exists
        if not client.schema.exists("CodebaseContent"):
            schema = Configure.configure_schema(
                class_name="CodebaseContent",
                properties=[
                    Property(name="content_type", data_type=DataType.TEXT),
                    Property(name="file_path", data_type=DataType.TEXT),
                    Property(name="language", data_type=DataType.TEXT),
                    Property(name="description", data_type=DataType.TEXT),
                    Property(name="timestamp", data_type=DataType.DATETIME),
                    Property(name="metadata", data_type=DataType.TEXT),
                ],
                vectorizer_config=Configure.Vectorizer.none(),
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric="cosine"
                )
            )
            
            client.schema.create(schema)
        
        logger.info("Weaviate client initialized with multimodal schema")
        return client
    
    async def index_multimodal_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """
        Index multimodal content into vector store
        
        Args:
            content: Dictionary with content data
                - 'text': str
                - 'images': List[bytes or str (base64)]
                - 'diagrams': List[dict]
                - 'audio': List[bytes]
                - 'metadata': dict
        
        Returns:
            Indexing results
        """
        start_time = asyncio.get_event_loop().time()
        
        results = {
            'text_chunks': 0,
            'image_embeddings': 0,
            'diagram_chunks': 0,
            'audio_transcriptions': 0,
            'total_vectors': 0,
            'processing_time': 0
        }
        
        try:
            # Process each content type
            if 'text' in content:
                text_results = await self._process_and_index_text(
                    content['text'], content.get('metadata', {})
                )
                results['text_chunks'] = text_results['chunks']
                results['total_vectors'] += text_results['vectors']
            
            if 'images' in content:
                image_results = await self._process_and_index_images(
                    content['images'], content.get('metadata', {})
                )
                results['image_embeddings'] = image_results['embeddings']
                results['total_vectors'] += image_results['vectors']
            
            if 'diagrams' in content:
                diagram_results = await self._process_and_index_diagrams(
                    content['diagrams'], content.get('metadata', {})
                )
                results['diagram_chunks'] = diagram_results['chunks']
                results['total_vectors'] += diagram_results['vectors']
            
            if 'audio' in content:
                audio_results = await self._process_and_index_audio(
                    content['audio'], content.get('metadata', {})
                )
                results['audio_transcriptions'] = audio_results['transcriptions']
                results['total_vectors'] += audio_results['vectors']
            
            processing_time = asyncio.get_event_loop().time() - start_time
            results['processing_time'] = round(processing_time, 2)
            
            logger.info(f"Indexed multimodal content: {results}")
            
            return {
                'status': 'success',
                'results': results,
                'message': f'Indexed {results["total_vectors"]} vectors in {results["processing_time"]}s'
            }
            
        except Exception as e:
            logger.error(f"Multimodal indexing failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'results': results
            }
    
    async def query_multimodal(self, 
                             text_query: str, 
                             image_query: Optional[Union[str, bytes]] = None,
                             filters: Optional[Dict] = None,
                             top_k: int = 5) -> Dict[str, Any]:
        """
        Execute multimodal query across all content types
        
        Args:
            text_query: Natural language query
            image_query: Optional image for visual search
            filters: Query filters
            top_k: Number of results to return
        
        Returns:
            Query results with ranked relevance
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Generate embeddings for query
            query_embeddings = await self._generate_query_embeddings(text_query, image_query)
            
            # Execute hybrid search across content types
            search_results = await self._execute_hybrid_search(
                query_embeddings, filters, top_k
            )
            
            # Generate response using retrieved context
            response = await self._generate_multimodal_response(
                text_query, search_results
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            
            return {
                'status': 'success',
                'query': text_query,
                'response': response,
                'sources': search_results,
                'processing_time': round(processing_time, 2),
                'confidence': self._calculate_query_confidence(search_results),
                'content_types': self._analyze_content_types(search_results)
            }
            
        except Exception as e:
            logger.error(f"Multimodal query failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'sources': []
            }
    
    async def _process_and_index_text(self, text: str, metadata: Dict) -> Dict[str, int]:
        """Process and index text content"""
        # Chunk text for optimal embedding
        chunks = self._chunk_text(text, chunk_size=1000, overlap=200)
        vectors_created = 0
        
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding = await self.text_embedder.aembed_query(chunk)
                
                # Store in Weaviate
                self.weaviate_client.data_object.create(
                    data_object={
                        'content_type': 'text',
                        'file_path': metadata.get('file_path', f'chunk_{i}'),
                        'language': metadata.get('language', 'unknown'),
                        'description': chunk[:200] + '...' if len(chunk) > 200 else chunk,
                        'timestamp': metadata.get('timestamp', 'now'),
                        'metadata': str(metadata)
                    },
                    class_name='CodebaseContent',
                    vector=embedding
                )
                
                vectors_created += 1
                
            except Exception as e:
                logger.warning(f"Failed to index text chunk {i}: {str(e)}")
                continue
        
        return {'chunks': len(chunks), 'vectors': vectors_created}
    
    async def _process_and_index_images(self, images: List[Union[str, bytes]], metadata: Dict) -> Dict[str, int]:
        """Process and index image content using CLIP"""
        embeddings_created = 0
        
        for i, image_data in enumerate(images):
            try:
                # Load and preprocess image
                if isinstance(image_data, str):  # base64
                    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                else:  # bytes
                    image = Image.open(io.BytesIO(image_data))
                
                # Generate multimodal embedding (simplified - would use CLIP)
                # In production: use actual CLIP model
                embedding = self._generate_image_embedding(image)
                
                # Store image metadata and embedding
                self.weaviate_client.data_object.create(
                    data_object={
                        'content_type': 'image',
                        'file_path': metadata.get('file_path', f'image_{i}'),
                        'language': 'visual',
                        'description': metadata.get('image_caption', f'Diagram {i+1}'),
                        'timestamp': metadata.get('timestamp', 'now'),
                        'metadata': str({
                            **metadata,
                            'image_dimensions': image.size,
                            'image_format': image.format
                        })
                    },
                    class_name='CodebaseContent',
                    vector=embedding
                )
                
                embeddings_created += 1
                
            except Exception as e:
                logger.warning(f"Failed to index image {i}: {str(e)}")
                continue
        
        return {'embeddings': embeddings_created, 'vectors': embeddings_created}
    
    async def _process_and_index_diagrams(self, diagrams: List[Dict], metadata: Dict) -> Dict[str, int]:
        """Process and index diagram content (UML, flowcharts)"""
        chunks_created = 0
        
        for i, diagram in enumerate(diagrams):
            try:
                # Extract text from diagram (OCR if image, parse if PlantUML/Mermaid)
                diagram_text = self._extract_diagram_text(diagram)
                
                # Generate embeddings for diagram elements
                elements = self._parse_diagram_elements(diagram_text)
                
                for element in elements[:5]:  # Limit per diagram
                    embedding = await self.text_embedder.aembed_query(element['description'])
                    
                    self.weaviate_client.data_object.create(
                        data_object={
                            'content_type': 'diagram',
                            'file_path': metadata.get('file_path', f'diagram_{i}'),
                            'language': 'uml' if 'plantuml' in diagram_text.lower() else 'mermaid',
                            'description': element['description'],
                            'timestamp': metadata.get('timestamp', 'now'),
                            'metadata': str({
                                **metadata,
                                **element,
                                'diagram_type': diagram.get('type', 'unknown'),
                                'relationships': element.get('relationships', [])
                            })
                        },
                        class_name='CodebaseContent',
                        vector=embedding
                    )
                    
                    chunks_created += 1
                
            except Exception as e:
                logger.warning(f"Failed to index diagram {i}: {str(e)}")
                continue
        
        return {'chunks': chunks_created, 'vectors': chunks_created}
    
    async def _process_and_index_audio(self, audio_files: List[bytes], metadata: Dict) -> Dict[str, int]:
        """Process and index audio content (transcription)"""
        transcriptions = 0
        
        for i, audio_data in enumerate(audio_files):
            try:
                # Transcribe audio (would use Whisper)
                transcription = self._transcribe_audio(audio_data)
                
                if transcription:
                    # Index transcription as text
                    text_results = await self._process_and_index_text(
                        transcription, {
                            **metadata,
                            'file_path': f'audio_{i}.wav',
                            'content_type': 'audio_transcription',
                            'speaker': metadata.get('speaker', 'unknown'),
                            'duration': self._get_audio_duration(audio_data)
                        }
                    )
                    
                    transcriptions += text_results['chunks']
                
            except Exception as e:
                logger.warning(f"Failed to process audio {i}: {str(e)}")
                continue
        
        return {'transcriptions': transcriptions, 'vectors': transcriptions}
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Chunk text for optimal embedding size"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size // 4):  # ~250 words per chunk
            chunk = ' '.join(words[i:i + chunk_size // 4])
            if len(chunk) > 20:  # Minimum length
                chunks.append(chunk)
        
        return chunks
    
    def _generate_image_embedding(self, image: Image.Image) -> List[float]:
        """Generate embedding for image (mock - use CLIP in production)"""
        # Mock implementation - replace with actual CLIP model
        import numpy as np
        
        # Simple hash-based embedding for demo
        img_array = np.array(image.resize((224, 224)))
        embedding = np.mean(img_array, axis=(0, 1, 2)).tolist()
        
        # Pad/truncate to standard dimension (1536 for OpenAI)
        while len(embedding) < 1536:
            embedding.append(0.0)
        embedding = embedding[:1536]
        
        return embedding
    
    def _extract_diagram_text(self, diagram: Dict) -> str:
        """Extract text content from diagram representation"""
        if diagram.get('type') == 'plantuml':
            return diagram.get('source', '')
        elif diagram.get('type') == 'mermaid':
            return diagram.get('code', '')
        else:
            return diagram.get('caption', '')
    
    def _parse_diagram_elements(self, diagram_text: str) -> List[Dict]:
        """Parse diagram into searchable elements"""
        elements = []
        
        # Simple parsing - extract classes, functions, relationships
        lines = diagram_text.split('\n')
        
        for line in lines:
            if 'class' in line.lower() or 'interface' in line.lower():
                elements.append({
                    'type': 'class',
                    'name': line.split()[-1].strip('{}'),
                    'description': line,
                    'relationships': []
                })
            elif '->' in line or '-->' in line:
                elements.append({
                    'type': 'relationship',
                    'description': line,
                    'relationships': line.split('->')
                })
        
        return elements[:10]
    
    def _transcribe_audio(self, audio_data: bytes) -> str:
        """Transcribe audio to text (mock - use Whisper)"""
        # Mock transcription
        return "This is a mock audio transcription from code review meeting."
    
    def _get_audio_duration(self, audio_data: bytes) -> float:
        """Get audio duration (mock)"""
        return 30.5  # seconds
    
    async def _generate_query_embeddings(self, text_query: str, image_query: Optional[Union[str, bytes]] = None) -> List[List[float]]:
        """Generate embeddings for multimodal query"""
        embeddings = []
        
        # Text embedding
        text_embedding = await self.text_embedder.aembed_query(text_query)
        embeddings.append(text_embedding)
        
        # Image embedding if provided
        if image_query:
            if isinstance(image_query, str):
                image = Image.open(io.BytesIO(base64.b64decode(image_query)))
            else:
                image = Image.open(io.BytesIO(image_query))
            
            image_embedding = self._generate_image_embedding(image)
            embeddings.append(image_embedding)
        
        return embeddings
    
    async def _execute_hybrid_search(self, query_embeddings: List[List[float]], 
                                   filters: Optional[Dict] = None, top_k: int = 5) -> List[Dict]:
        """Execute hybrid search across all content types"""
        results = []
        
        # Search each content type
        content_types = ['text', 'image', 'diagram']
        
        for content_type in content_types:
            try:
                type_results = await self._search_content_type(
                    content_type, query_embeddings, filters, top_k // len(content_types)
                )
                results.extend(type_results)
            except Exception as e:
                logger.warning(f"Search failed for {content_type}: {str(e)}")
                continue
        
        # Rank and filter results
        ranked_results = self._rank_and_filter_results(results, top_k)
        
        return ranked_results
    
    async def _search_content_type(self, content_type: str, embeddings: List[List[float]], 
                                 filters: Dict, limit: int) -> List[Dict]:
        """Search specific content type"""
        query = (
            self.weaviate_client.query
            .get("CodebaseContent", ["content_type", "file_path", "description", "metadata"])
            .with_where({
                "path": ["content_type"],
                "operator": "Equal",
                "valueText": content_type
            })
            .with_near_vector({
                "vector": embeddings[0][:1536],  # Use first embedding
                "certainty": 0.7
            })
            .with_limit(limit)
            .do()
        )
        
        results = []
        for item in query['data']['Get']['CodebaseContent']:
            results.append({
                'content_type': item['content_type'],
                'file_path': item['file_path'],
                'description': item['description'],
                'metadata': item['metadata'],
                'relevance_score': 0.8  # Mock - would calculate actual similarity
            })
        
        return results
    
    def _rank_and_filter_results(self, results: List[Dict], top_k: int) -> List[Dict]:
        """Rank results by relevance and filter"""
        # Simple ranking - in production use more sophisticated algorithm
        ranked = sorted(results, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return ranked[:top_k]
    
    async def _generate_multimodal_response(self, query: str, sources: List[Dict]) -> str:
        """Generate response using multimodal context"""
        # Create rich context from multiple sources
        context_parts = []
        
        for source in sources[:3]:  # Top 3 sources
            if source['content_type'] == 'text':
                context_parts.append(f"From {source['file_path']}:")
                context_parts.append(source['description'][:500])
            elif source['content_type'] == 'image':
                context_parts.append(f"From diagram/image {source['file_path']}:")
                context_parts.append(source['description'])
            elif source['content_type'] == 'diagram':
                context_parts.append(f"From {source['content_type']} {source['file_path']}:")
                context_parts.append(source['description'])
        
        context = '\n\n'.join(context_parts)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert code assistant with access to multimodal codebase information.
            
            Use the provided context from text, diagrams, and other sources to answer the user's question.
            
            Be specific and reference exact files, line numbers, or diagram elements when possible.
            If the context includes visual information, describe relationships and structures clearly.
            
            Structure your response:
            1. Direct answer to the question
            2. Relevant code snippets or diagram descriptions
            3. Implementation details or architecture explanation
            4. Any assumptions made based on available context
            """),
            ("user", """Context from multimodal codebase:
            
            {context}
            
            Question: {query}
            
            Provide a comprehensive answer using all available context sources."""),
        ])
        
        chain = prompt | self.llm
        response = await chain.ainvoke({
            "context": context,
            "query": query
        })
        
        return response.content
    
    def _calculate_query_confidence(self, sources: List[Dict]) -> float:
        """Calculate confidence score for query results"""
        if not sources:
            return 0.0
        
        avg_relevance = sum(s.get('relevance_score', 0) for s in sources) / len(sources)
        diversity_bonus = len(set(s['content_type'] for s in sources)) * 0.1
        
        return min(1.0, avg_relevance + diversity_bonus)
    
    def _analyze_content_types(self, sources: List[Dict]) -> Dict[str, int]:
        """Analyze distribution of content types in results"""
        type_counts = {}
        
        for source in sources:
            content_type = source.get('content_type', 'unknown')
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        
        return type_counts


# Privacy and Governance Layer
class PrivacyGovernanceLayer:
    """Data privacy and compliance layer"""
    
    SECRET_PATTERNS = [
        r'sk-[a-zA-Z0-9]{48}',  # OpenAI API keys
        r'api_key\s*=\s*[\"\'][^\"\']+[\"\']',  # API keys
        r'password\s*=\s*[\"\'][^\"\']+[\"\']',  # Passwords
        r'token\s*=\s*[\"\'][^\"\']+[\"\']',  # Tokens
        r'aws_access_key_id\s*=\s*[\"\'][^\"\']+[\"\']',  # AWS keys
        r'private_key\s*=\s*[\"\'][^\"\']+[\"\']',  # Private keys
        r'\b[A-Fa-f0-9]{32}\b',  # MD5 hashes (potential secrets)
        r'\b[A-Fa-f0-9]{40}\b',  # SHA1 hashes
        r'\b[A-Fa-f0-9]{64}\b',  # SHA256 hashes
    ]
    
    COMPLIANCE_RULES = {
        'gdpr': {
            'pii_patterns': [
                r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
                r'\b[A-Za-z]+\s[A-Za-z]+\s\d{4}\b',  # Names with years
                r'\b\d{3}-\d{3}-\d{4}\b',  # Phone numbers
            ],
            'action': 'redact'
        },
        'hipaa': {
            'medical_patterns': [
                r'\b[A-Za-z]+\s+(Dr|MD|DO|NP|PA)\b',  # Medical professionals
                r'\b\d{6}\d{2}\d{4}\b',  # Medical record numbers
            ],
            'action': 'alert'
        }
    }
    
    def __init__(self):
        self.audit_log = []
        self.redaction_count = 0
        
    def process_content(self, content: str, user_id: str, operation: str, 
                       compliance_mode: str = 'strict') -> Dict[str, Any]:
        """
        Process content through privacy filters
        
        Args:
            content: Text content to process
            user_id: User identifier
            operation: Type of operation (analysis, storage, etc.)
            compliance_mode: 'strict', 'moderate', 'lenient'
        
        Returns:
            Processed content with audit information
        """
        start_time = time.time()
        
        processed_content = content
        findings = []
        
        # 1. Secret detection and redaction
        secret_findings, processed_content = self._detect_and_redact_secrets(processed_content)
        findings.extend(secret_findings)
        
        # 2. PII and compliance checks
        compliance_findings = self._check_compliance(processed_content, compliance_mode)
        findings.extend(compliance_findings)
        
        # 3. Audit logging
        audit_entry = self._create_audit_entry(
            user_id, operation, processed_content, findings
        )
        self.audit_log.append(audit_entry)
        
        processing_time = time.time() - start_time
        
        return {
            'processed_content': processed_content,
            'findings': findings,
            'redactions_made': len(secret_findings),
            'compliance_issues': len([f for f in findings if f['type'] == 'compliance']),
            'processing_time': round(processing_time, 3),
            'audit_id': audit_entry['id'],
            'risk_score': self._calculate_risk_score(findings)
        }
    
    def _detect_and_redact_secrets(self, content: str) -> tuple[List[Dict], str]:
        """Detect and redact secrets from content"""
        findings = []
        processed = content
        
        for i, pattern in enumerate(self.SECRET_PATTERNS):
            matches = list(re.finditer(pattern, processed))
            
            for match in matches:
                start, end = match.span()
                original = match.group()
                redacted = '[SECRET_REDACTED]'
                
                processed = (processed[:start] + redacted + 
                           processed[end:])
                
                # Adjust positions for subsequent matches
                for later_match in matches:
                    if later_match.start() > end:
                        later_match_start = later_match.start() - (end - start - len(redacted))
                        later_match_end = later_match.end() - (end - start - len(redacted))
                        later_match.span = lambda: (later_match_start, later_match_end)
                
                findings.append({
                    'type': 'secret',
                    'pattern_id': i,
                    'original_position': (start, end),
                    'redacted_length': len(redacted),
                    'original_value': original[:10] + '...' if len(original) > 10 else original,
                    'risk_level': 'high',
                    'category': self._classify_secret(original)
                })
                
                self.redaction_count += 1
        
        return findings, processed
    
    def _check_compliance(self, content: str, mode: str) -> List[Dict]:
        """Check content against compliance rules"""
        findings = []
        
        for standard, rules in self.COMPLIANCE_RULES.items():
            if mode == 'lenient' and standard == 'hipaa':
                continue  # Skip strict checks in lenient mode
            
            for rule_name, pattern in enumerate(rules['pii_patterns']):
                matches = list(re.finditer(pattern, content))
                
                for match in matches:
                    findings.append({
                        'type': 'compliance',
                        'standard': standard,
                        'rule': rule_name,
                        'action': rules['action'],
                        'position': match.span(),
                        'matched_text': match.group(),
                        'risk_level': 'medium' if rules['action'] == 'alert' else 'high',
                        'recommendation': f"Review {standard} compliance for {rules['action']} required"
                    })
        
        return findings
    
    def _create_audit_entry(self, user_id: str, operation: str, content: str, 
                           findings: List[Dict]) -> Dict:
        """Create audit log entry"""
        audit_id = f"audit_{int(time.time()*1000)}_{hash(user_id) % 10000}"
        
        entry = {
            'id': audit_id,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'operation': operation,
            'content_length': len(content),
            'findings_count': len(findings),
            'secrets_redacted': len([f for f in findings if f['type'] == 'secret']),
            'compliance_issues': len([f for f in findings if f['type'] == 'compliance']),
            'risk_score': self._calculate_risk_score(findings),
            'content_hash': hashlib.sha256(content.encode()).hexdigest(),
            'ip_address': self._get_client_ip(),  # Implement based on request context
            'user_agent': self._get_user_agent(),  # Implement based on request
        }
        
        # Store audit log (in production: use database)
        self._persist_audit_log(entry)
        
        return entry
    
    def _calculate_risk_score(self, findings: List[Dict]) -> float:
        """Calculate overall risk score"""
        if not findings:
            return 0.0
        
        risk_weights = {
            'secret': 0.4,
            'compliance': 0.3,
            'high': 0.2,
            'medium': 0.1,
            'low': 0.0
        }
        
        total_risk = 0.0
        for finding in findings:
            weight = risk_weights.get(finding['type'], 0.1)
            level_weight = risk_weights.get(finding['risk_level'], 0.1)
            total_risk += weight * level_weight
        
        return round(total_risk * 100, 1)  # Scale to 0-100
    
    def _classify_secret(self, secret: str) -> str:
        """Classify type of detected secret"""
        secret_lower = secret.lower()
        
        if 'sk-' in secret_lower:
            return 'api_key'
        elif 'password' in secret_lower:
            return 'password'
        elif 'token' in secret_lower:
            return 'token'
        elif any(key in secret_lower for key in ['aws', 'azure', 'gcp']):
            return 'cloud_key'
        elif len(secret) in [32, 40, 64] and all(c in '0123456789abcdefABCDEF' for c in secret):
            return 'hash_key'
        else:
            return 'unknown_secret'
    
    def _get_client_ip(self) -> str:
        """Get client IP (mock - implement based on request context)"""
        return "127.0.0.1"
    
    def _get_user_agent(self) -> str:
        """Get user agent (mock)"""
        return "CodeMind-AI-Client/1.0"
    
    def _persist_audit_log(self, entry: Dict):
        """Persist audit log entry (mock - use database)"""
        # In production: insert into audit database
        logger.info(f"Audit log entry created: {entry['id']}")
        
        # Keep recent entries in memory
        if len(self.audit_log) > 1000:
            self.audit_log.pop(0)
    
    def get_audit_report(self, user_id: str, time_range: tuple = None) -> List[Dict]:
        """Generate audit report for user"""
        if time_range:
            start, end = time_range
            filtered = [e for e in self.audit_log 
                       if e['user_id'] == user_id 
                       and start <= e['timestamp'] <= end]
        else:
            filtered = [e for e in self.audit_log if e['user_id'] == user_id]
        
        return {
            'user_id': user_id,
            'total_operations': len(filtered),
            'high_risk_operations': len([e for e in filtered if e['risk_score'] > 70]),
            'secrets_redacted': sum(e['secrets_redacted'] for e in filtered),
            'entries': filtered[-50:],  # Last 50 entries
            'generated_at': datetime.now().isoformat()
        }
    
    def export_compliance_report(self, format: str = 'json') -> str:
        """Export compliance report"""
        report = {
            'total_redactions': self.redaction_count,
            'audit_entries': len(self.audit_log),
            'high_risk_operations': len([e for e in self.audit_log if e['risk_score'] > 70]),
            'compliance_standards': list(self.COMPLIANCE_RULES.keys()),
            'generated_at': datetime.now().isoformat()
        }
        
        if format == 'json':
            return json.dumps(report, indent=2)
        elif format == 'csv':
            return self._report_to_csv(report)
        else:
            raise ValueError(f"Unsupported format: {format}")


# Plugin Architecture
class PluginManager:
    """Advanced plugin system for extensibility"""
    
    def __init__(self, plugin_dir: str = 'plugins'):
        self.plugins = {}
        self.plugin_dir = plugin_dir
        self.load_plugins()
    
    def load_plugins(self):
        """Load available plugins from directory"""
        try:
            import importlib.util
            import os
            
            plugin_files = [f for f in os.listdir(self.plugin_dir) 
                          if f.endswith('.py') and not f.startswith('_')]
            
            for plugin_file in plugin_files:
                plugin_name = plugin_file[:-3]
                spec = importlib.util.spec_from_file_location(
                    plugin_name, f"{self.plugin_dir}/{plugin_file}"
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for plugin classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, BasePlugin) and 
                        attr != BasePlugin):
                        
                        plugin_instance = attr()
                        self.plugins[plugin_name] = plugin_instance
                        logger.info(f"Loaded plugin: {plugin_name}")
            
        except Exception as e:
            logger.error(f"Failed to load plugins: {str(e)}")
    
    async def execute_plugin(self, plugin_name: str, task: Dict, context: Dict = None) -> Dict:
        """Execute specific plugin with task"""
        if plugin_name not in self.plugins:
            raise ValueError(f"Plugin '{plugin_name}' not found. Available: {list(self.plugins.keys())}")
        
        plugin = self.plugins[plugin_name]
        
        try:
            result = await plugin.execute(task, context or {})
            
            # Validate plugin response
            if not isinstance(result, dict) or 'status' not in result:
                logger.warning(f"Plugin {plugin_name} returned invalid response format")
                return {'status': 'error', 'message': 'Invalid plugin response'}
            
            return {
                'status': 'success',
                'plugin': plugin_name,
                'version': plugin.version,
                'result': result,
                'execution_time': getattr(result, 'execution_time', 0),
                'metadata': plugin.metadata
            }
            
        except Exception as e:
            logger.error(f"Plugin {plugin_name} execution failed: {str(e)}")
            return {
                'status': 'error',
                'plugin': plugin_name,
                'error': str(e),
                'metadata': {'error_type': type(e).__name__}
            }
    
    def get_available_plugins(self) -> List[Dict]:
        """Get metadata for all available plugins"""
        plugins_info = []
        
        for name, plugin in self.plugins.items():
            plugins_info.append({
                'name': name,
                'version': plugin.version,
                'description': plugin.description,
                'capabilities': plugin.capabilities,
                'dependencies': plugin.dependencies,
                'author': plugin.author,
                'status': 'active'
            })
        
        return sorted(plugins_info, key=lambda x: x['name'])
    
    def validate_plugin_task(self, plugin_name: str, task: Dict) -> Dict[str, Any]:
        """Validate task compatibility with plugin"""
        if plugin_name not in self.plugins:
            return {'valid': False, 'error': f"Plugin '{plugin_name}' not found"}
        
        plugin = self.plugins[plugin_name]
        required_fields = plugin.required_task_fields
        
        missing_fields = [field for field in required_fields if field not in task]
        
        if missing_fields:
            return {
                'valid': False,
                'error': f'Missing required fields: {missing_fields}',
                'required': required_fields
            }
        
        # Validate field types (simplified)
        type_errors = []
        for field, expected_type in plugin.task_field_types.items():
            if field in task and not isinstance(task[field], expected_type):
                type_errors.append(f"'{field}' should be {expected_type.__name__}")
        
        if type_errors:
            return {
                'valid': False,
                'error': 'Type validation failed',
                'details': type_errors
            }
        
        return {'valid': True, 'plugin': plugin_name}


class BasePlugin:
    """Base class for all plugins"""
    
    version = "1.0.0"
    description = "Base plugin template"
    capabilities = []
    dependencies = []
    author = "CodeMind AI Team"
    required_task_fields = []
    task_field_types = {}
    metadata = {}
    
    async def execute(self, task: Dict, context: Dict) -> Dict:
        """Execute plugin task - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement execute() method")


# Example: GitHub Actions Plugin
class GitHubActionsPlugin(BasePlugin):
    """Plugin for GitHub Actions integration"""
    
    version = "1.2.0"
    description = "Integrate with GitHub Actions for CI/CD automation"
    capabilities = ['pr_review', 'commit_analysis', 'release_automation']
    dependencies = ['PyGitHub>=1.58']
    author = "CodeMind AI Team"
    
    required_task_fields = ['repository', 'action_type']
    task_field_types = {
        'repository': str,
        'action_type': str,
        'pr_number': (int, type(None)),
        'commit_sha': (str, type(None))
    }
    
    def __init__(self):
        super().__init__()
        from github import Github
        self.gh = Github(settings.GITHUB_TOKEN)
        
    async def execute(self, task: Dict, context: Dict) -> Dict:
        """Execute GitHub Actions task"""
        action_type = task.get('action_type')
        
        start_time = time.time()
        
        try:
            if action_type == 'pr_review':
                result = await self._review_pull_request(task, context)
            elif action_type == 'commit_analysis':
                result = await self._analyze_commit(task, context)
            elif action_type == 'release_automation':
                result = await self._automate_release(task, context)
            else:
                return {
                    'status': 'error',
                    'message': f'Unknown action type: {action_type}',
                    'supported': self.capabilities
                }
            
            execution_time = time.time() - start_time
            result['execution_time'] = round(execution_time, 2)
            
            return {
                'status': 'success',
                'action': action_type,
                'result': result,
                'metadata': {
                    'plugin_version': self.version,
                    'api_calls': getattr(result, 'api_calls', 1),
                    'changes_made': getattr(result, 'changes_made', 0)
                }
            }
            
        except Exception as e:
            logger.error(f"GitHub plugin failed: {str(e)}")
            return {
                'status': 'error',
                'action': action_type,
                'error': str(e),
                'metadata': {'error_type': type(e).__name__}
            }
    
    async def _review_pull_request(self, task: Dict, context: Dict) -> Dict:
        """AI-powered pull request review"""
        repo_name = task['repository']
        pr_number = task.get('pr_number')
        
        repo = self.gh.get_repo(repo_name)
        pr = repo.get_pull(pr_number)
        
        # Get changed files
        files = pr.get_files()
        code_changes = []
        
        for file in files:
            if file.status in ['added', 'modified']:
                code_changes.append({
                    'filename': file.filename,
                    'patch': file.patch,
                    'additions': file.additions,
                    'deletions': file.deletions
                })
        
        # Analyze changes with AI agents
        analysis_results = []
        
        for change in code_changes[:5]:  # Limit for performance
            if change['patch']:
                # Mock AI analysis - integrate with actual agents
                analysis = await self._analyze_code_change(change['patch'], change['filename'])
                analysis_results.append(analysis)
        
        # Generate review comments
        review_comments = self._generate_review_comments(analysis_results)
        
        # Post comments to PR
        comments_posted = 0
        for comment in review_comments[:10]:  # GitHub limit
            try:
                pr.create_review_comment(
                    body=comment['text'],
                    commit=pr.head.sha,
                    path=comment['file'],
                    position=1  # Simplified
                )
                comments_posted += 1
            except Exception as e:
                logger.warning(f"Failed to post comment: {str(e)}")
        
        return {
            'pr_number': pr_number,
            'files_analyzed': len(code_changes),
            'issues_found': sum(len(r.get('findings', [])) for r in analysis_results),
            'comments_posted': comments_posted,
            'overall_score': self._calculate_pr_score(analysis_results),
            'api_calls': 5  # Approximate
        }
    
    async def _analyze_code_change(self, patch: str, filename: str) -> Dict:
        """Analyze individual code change"""
        # Integrate with CodeMind AI agents
        # Mock implementation
        language = filename.split('.')[-1]
        
        return {
            'filename': filename,
            'language': language,
            'findings': [
                {
                    'severity': 'medium',
                    'description': 'Consider adding error handling',
                    'line': 1,
                    'suggestion': 'Add try-catch block'
                }
            ],
            'score': 85,
            'recommendations': ['Add tests', 'Update documentation']
        }
    
    def _generate_review_comments(self, analysis_results: List[Dict]) -> List[Dict]:
        """Generate review comments from analysis"""
        comments = []
        
        for analysis in analysis_results:
            for finding in analysis.get('findings', []):
                comments.append({
                    'file': analysis['filename'],
                    'line': finding.get('line', 1),
                    'text': f"**{finding['severity'].title()}**: {finding['description']}\n\n" +
                           f"**Suggestion**: {finding.get('suggestion', 'Review this change')}\n\n" +
                           f"**Score Impact**: {analysis.get('score', 0)}/100",
                    'priority': finding['severity']
                })
        
        return sorted(comments, key=lambda x: x['priority'], reverse=True)
    
    def _calculate_pr_score(self, analysis_results: List[Dict]) -> float:
        """Calculate overall PR quality score"""
        total_score = 100.0
        issue_penalty = 0.0
        
        for analysis in analysis_results:
            findings = analysis.get('findings', [])
            severity_weights = {'critical': 20, 'high': 10, 'medium': 5, 'low': 1}
            
            for finding in findings:
                weight = severity_weights.get(finding['severity'], 2)
                issue_penalty += weight
            
            total_score -= issue_penalty
        
        return round(max(0, total_score), 1)
    
    async def _analyze_commit(self, task: Dict, context: Dict) -> Dict:
        """Analyze commit for quality and security"""
        # Implementation for commit analysis
        pass
    
    async def _automate_release(self, task: Dict, context: Dict) -> Dict:
        """Automate release process"""
        # Implementation for release automation
        pass


# MLOps Integration
class MLOpsManager:
    """MLOps integration for model management and monitoring"""
    
    def __init__(self):
        self.model_registry = {}
        self.experiment_tracker = []
        self.performance_metrics = {}
    
    async def register_model(self, model_name: str, model_config: Dict, 
                           version: str, metadata: Dict) -> str:
        """Register new model version"""
        model_id = f"{model_name}_v{version.replace('.', '_')}"
        
        registration = {
            'id': model_id,
            'name': model_name,
            'version': version,
            'config': model_config,
            'metadata': metadata,
            'registered_at': datetime.now().isoformat(),
            'status': 'active',
            'usage_count': 0,
            'performance_score': 0.0
        }
        
        self.model_registry[model_id] = registration
        
        logger.info(f"Registered model: {model_id}")
        return model_id
    
    async def log_experiment(self, experiment: Dict) -> str:
        """Log A/B experiment results"""
        experiment_id = f"exp_{int(time.time()*1000)}"
        
        experiment['id'] = experiment_id
        experiment['timestamp'] = datetime.now().isoformat()
        experiment['status'] = 'completed'
        
        self.experiment_tracker.append(experiment)
        
        # Keep only recent experiments
        if len(self.experiment_tracker) > 1000:
            self.experiment_tracker.pop(0)
        
        return experiment_id
    
    async def track_performance(self, model_id: str, metrics: Dict) -> Dict:
        """Track model performance metrics"""
        if model_id not in self.model_registry:
            return {'status': 'error', 'message': 'Model not found'}
        
        # Update usage count
        self.model_registry[model_id]['usage_count'] += 1
        
        # Calculate rolling performance score
        current_score = self.model_registry[model_id].get('performance_score', 0.0)
        new_score = (current_score * 0.9 + metrics.get('accuracy', 0) * 0.1)
        
        self.model_registry[model_id]['performance_score'] = round(new_score, 3)
        self.model_registry[model_id]['last_used'] = datetime.now().isoformat()
        
        # Store detailed metrics
        timestamp = datetime.now().isoformat()
        self.performance_metrics[f"{model_id}_{timestamp}"] = {
            **metrics,
            'timestamp': timestamp,
            'model_id': model_id
        }
        
        # Cleanup old metrics
        cutoff_time = datetime.now() - timedelta(days=30)
        self.performance_metrics = {
            k: v for k, v in self.performance_metrics.items()
            if datetime.fromisoformat(v['timestamp']) > cutoff_time
        }
        
        return {
            'status': 'success',
            'model_id': model_id,
            'updated_score': self.model_registry[model_id]['performance_score'],
            'usage_count': self.model_registry[model_id]['usage_count']
        }
    
    async def get_model_leaderboard(self, limit: int = 10) -> List[Dict]:
        """Get ranked list of models by performance"""
        ranked_models = sorted(
            self.model_registry.values(),
            key=lambda x: x['performance_score'],
            reverse=True
        )
        
        return ranked_models[:limit]
    
    async def recommend_model(self, task_type: str, requirements: Dict) -> Optional[str]:
        """Recommend best model for specific task"""
        candidates = []
        
        for model_id, model_info in self.model_registry.items():
            if task_type in model_info.get('capabilities', []):
                # Score based on requirements
                score = self._calculate_model_fit_score(model_info, requirements)
                candidates.append((model_id, score))
        
        if candidates:
            best_model = max(candidates, key=lambda x: x[1])[0]
            return best_model
        
        return None
    
    def _calculate_model_fit_score(self, model_info: Dict, requirements: Dict) -> float:
        """Calculate how well model fits task requirements"""
        base_score = model_info.get('performance_score', 0.0)
        
        # Latency requirement
        if 'max_latency' in requirements:
            model_latency = model_info.get('avg_latency', 2.0)
            if model_latency > requirements['max_latency']:
                base_score *= 0.7
        
        # Accuracy requirement
        if 'min_accuracy' in requirements:
            if model_info.get('performance_score', 0) < requirements['min_accuracy']:
                base_score *= 0.8
        
        # Cost sensitivity
        if requirements.get('cost_sensitive', False):
            base_score *= model_info.get('cost_efficiency', 1.0)
        
        return base_score


# Integration example
class AdvancedCodeMindAPI:
    """Main API class integrating all advanced features"""
    
    def __init__(self):
        self.swarm_orchestrator = RaySwarmOrchestrator()
        self.multimodal_rag = MultimodalRAG()
        self.privacy_layer = PrivacyGovernanceLayer()
        self.plugin_manager = PluginManager()
        self.mlops_manager = MLOpsManager()
    
    async def advanced_analyze(self, code: str, user_id: str, 
                              context: Dict, plugins: List[str] = None) -> Dict:
        """Execute advanced analysis with all features"""
        
        # 1. Privacy processing
        privacy_result = self.privacy_layer.process_content(
            code, user_id, 'code_analysis', 'strict'
        )
        
        if privacy_result['risk_score'] > 80:
            return {
                'status': 'blocked',
                'reason': 'High risk content detected',
                'risk_score': privacy_result['risk_score'],
                'recommendations': ['Review redacted secrets', 'Check compliance issues']
            }
        
        clean_code = privacy_result['processed_content']
        
        # 2. Swarm analysis
        task = AgentTask(
            code=clean_code,
            language=context.get('language', 'python'),
            context=context
        )
        
        swarm_result = await self.swarm_orchestrator.execute_swarm_analysis(task)
        
        # 3. Plugin execution (if requested)
        plugin_results = {}
        if plugins:
            for plugin_name in plugins:
                validation = self.plugin_manager.validate_plugin_task(plugin_name, context)
                if validation['valid']:
                    plugin_result = await self.plugin_manager.execute_plugin(
                        plugin_name, context, {'analysis': swarm_result}
                    )
                    plugin_results[plugin_name] = plugin_result
        
        # 4. MLOps tracking
        model_used = 'swarm_v1.0'
        performance_metrics = {
            'accuracy': swarm_result['overall_score'] / 100,
            'latency': swarm_result['execution_time'],
            'issues_detected': len(swarm_result.get('coordinated_results', [])),
            'user_satisfaction': 0.0  # Would come from feedback
        }
        
        await self.mlops_manager.track_performance(model_used, performance_metrics)
        
        # 5. Final coordinated response
        final_result = {
            'status': 'success',
            'privacy': privacy_result,
            'swarm_analysis': swarm_result,
            'plugins': plugin_results,
            'mlops': {
                'model_used': model_used,
                'performance_tracked': True
            },
            'timestamp': datetime.now().isoformat(),
            'audit_id': privacy_result['audit_id'],
            'recommendations': self._prioritize_recommendations(
                swarm_result['recommendations'], 
                plugin_results
            )
        }
        
        return final_result
    
    async def multimodal_search(self, query: Dict, user_id: str) -> Dict:
        """Execute multimodal search with privacy controls"""
        
        # Privacy check on query
        privacy_result = self.privacy_layer.process_content(
            query.get('text', ''), user_id, 'search_query'
        )
        
        if privacy_result['risk_score'] > 50:
            return {'status': 'blocked', 'reason': 'Query contains sensitive information'}
        
        # Execute multimodal RAG
        rag_result = await self.multimodal_rag.query_multimodal(
            text_query=query.get('text', ''),
            image_query=query.get('image'),
            filters=query.get('filters', {}),
            top_k=query.get('top_k', 5)
        )
        
        # Track search performance
        await self.mlops_manager.track_performance(
            'multimodal_rag_v1',
            {
                'relevance_score': rag_result.get('confidence', 0),
                'results_returned': len(rag_result.get('sources', [])),
                'diversity': len(rag_result.get('content_types', {})),
                'latency': rag_result.get('processing_time', 0)
            }
        )
        
        return {
            'status': 'success',
            'privacy_check': privacy_result,
            'search_results': rag_result,
            'audit_id': privacy_result['audit_id']
        }
    
    def _prioritize_recommendations(self, swarm_recs: List[str], 
                                  plugin_results: Dict) -> List[Dict]:
        """Prioritize and deduplicate recommendations"""
        prioritized = []
        seen_recommendations = set()
        
        # Add swarm recommendations
        for rec in swarm_recs[:5]:
            rec_hash = hash(rec.lower())
            if rec_hash not in seen_recommendations:
                seen_recommendations.add(rec_hash)
                prioritized.append({
                    'text': rec,
                    'source': 'swarm',
                    'priority': 'high',
                    'type': 'general'
                })
        
        # Add plugin recommendations
        for plugin_name, result in plugin_results.items():
            if result['status'] == 'success' and 'recommendations' in result['result']:
                for rec in result['result']['recommendations'][:3]:
                    rec_hash = hash(str(rec).lower())
                    if rec_hash not in seen_recommendations:
                        seen_recommendations.add(rec_hash)
                        prioritized.append({
                            'text': str(rec),
                            'source': plugin_name,
                            'priority': 'medium',
                            'type': 'plugin',
                            'plugin_action': result['result'].get('action', '')
                        })
        
        return prioritized[:10]


# Usage example
async def demonstrate_advanced_features():
    """Demonstrate all advanced features"""
    api = AdvancedCodeMindAPI()
    
    # Example code with secrets (will be redacted)
    sample_code = """
    API_KEY = 'sk-1234567890abcdef'
def login(user):
    query = f"SELECT * FROM users WHERE name = '{user}'"
    return db.execute(query)
    """
    
    user_id = "demo_user_123"
    context = {
        'language': 'python',
        'project_type': 'web_app',
        'framework': 'fastapi',
        'needs_testing': True
    }
    
    # Advanced analysis
    print("=== Advanced Swarm Analysis ===")
    analysis_result = await api.advanced_analyze(
        sample_code, user_id, context, 
        plugins=['github_actions']
    )
    
    print(f"Privacy redacted: {analysis_result['privacy']['redactions_made']}")
    print(f"Overall score: {analysis_result['swarm_analysis']['overall_score']}")
    print(f"Recommendations: {len(analysis_result['recommendations'])}")
    
    # Multimodal search
    print("\n=== Multimodal Search ===")
    search_result = await api.multimodal_search({
        'text': 'Show authentication flow diagram',
        'top_k': 3
    }, user_id)
    
    print(f"Search confidence: {search_result['search_results']['confidence']:.2f}")
    print(f"Content types found: {search_result['search_results']['content_types']}")
    
    # MLOps leaderboard
    print("\n=== Model Leaderboard ===")
    leaderboard = await api.mlops_manager.get_model_leaderboard()
    for model in leaderboard[:3]:
        print(f"{model['name']} v{model['version']}: {model['performance_score']:.1f}")

if __name__ == "__main__":
    asyncio.run(demonstrate_advanced_features())
