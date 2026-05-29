from src.database import get_vectorstore

class ContractRetriever:
    """
    Handles semantic retrieval from the vector store with custom 
    filtering and lexical re-ranking logic.
    """

    def __init__(self):
        self.vectorstore = get_vectorstore()

    def search(
        self,
        question: str,
        k: int = 5,
        score_threshold: float = 1.2
    ):
        """
        Performs a FAISS similarity search followed by quality filtering
        and a hybrid lexical re-ranking strategy.
        """

        try:
            # Retrieve a larger pool of candidates to allow for effective filtering
            results = self.vectorstore.similarity_search_with_score(
                query=question,
                k=k * 5, 
            )

        except Exception as e:
            print(f"❌ Retrieval error: {e}")
            return []

        print("\n================ RETRIEVAL DEBUG ================")

        filtered_docs = []
        seen_contents = set()

        for doc, score in results:
            try:
                content = doc.page_content.strip()
                source = doc.metadata.get("source", "unknown")

                print(f"\n📄 SOURCE : {source}")
                print(f"🎯 SCORE  : {score}")
                print(content[:250])
                print("-" * 60)

                # Quality filtering: Ignore empty or overly short chunks
                if not content or len(content) < 120:
                    continue

                # Ignore documents that exceed the semantic distance threshold
                if score > score_threshold:
                    continue

                # De-duplication based on content similarity
                normalized = content[:300].lower()
                if normalized in seen_contents:
                    continue

                seen_contents.add(normalized)
                doc.metadata["retrieval_score"] = float(score)
                filtered_docs.append(doc)

            except Exception as e:
                print(f"⚠️ Chunk filtering error: {e}")

        # Fallback mechanism: if filtering is too aggressive, return top results
        if not filtered_docs:
            print("⚠️ No docs survived filtering -> fallback mode")
            fallback = []
            for doc, score in results[:2]:
                if len(doc.page_content.strip()) > 120:
                    doc.metadata["retrieval_score"] = float(score)
                    fallback.append(doc)
            filtered_docs = fallback

        # Initial sort based on semantic distance (retrieval_score)
        filtered_docs = sorted(
            filtered_docs,
            key=lambda d: d.metadata.get("retrieval_score", 999)
        )
        
        # Hybrid Re-ranking: Combine semantic score with lexical overlap (keyword matching)
        query_words = set(question.lower().split())

        def lexical_score(doc):
            content_words = set(doc.page_content.lower().split())
            return len(query_words.intersection(content_words))

        filtered_docs = sorted(
            filtered_docs,
            key=lambda d: (
                lexical_score(d),
                -d.metadata.get("retrieval_score", 999)
            ),
            reverse=True
        )
        
        filtered_docs = filtered_docs[:k]

        print(f"\n✅ FINAL FILTERED DOCS: {len(filtered_docs)}")
        return filtered_docs
    
    def search_by_source(self, source_name: str, limit: int = 10):
        """
        Retrieves chunks filtered strictly by a specific document source name.
        """
        try:
            docs = self.vectorstore.similarity_search(source_name, k=50)

            filtered = [doc for doc in docs if doc.metadata.get("source") == source_name]
            return filtered[:limit]

        except Exception as e:
            print(f"❌ Source search error: {e}")
            return []