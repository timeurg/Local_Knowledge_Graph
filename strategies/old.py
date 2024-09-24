
from chat.get_short_title import get_short_title
from db.embeddings import EmbeddingDB
import time
import json
from chat.api import get_embedding, stream_api_call
from db.embeddings import get_db, EmbeddingDB
from graph.helpers import calculate_strongest_path, serialize_graph_data
from helpers import calculate_top_similarities, extract_json


def generate_response(prompt, conn: EmbeddingDB):
    messages = [
        {"role": "system", "content": """You are an expert AI assistant that explains your reasoning step by step. For each step, provide a title that describes what you're doing in that step, along with the content. Decide if you need another step or if you're ready to give the final answer. Respond in JSON format with 'title', 'content', and 'next_action' (either 'continue' or 'final_answer') keys. USE AS MANY REASONING STEPS AS POSSIBLE. AT LEAST 3. BE AWARE OF YOUR LIMITATIONS AS AN LLM AND WHAT YOU CAN AND CANNOT DO. IN YOUR REASONING, INCLUDE EXPLORATION OF ALTERNATIVE ANSWERS. CONSIDER YOU MAY BE WRONG, AND IF YOU ARE WRONG IN YOUR REASONING, WHERE IT WOULD BE. FULLY TEST ALL OTHER POSSIBILITIES. YOU CAN BE WRONG. WHEN YOU SAY YOU ARE RE-EXAMINING, ACTUALLY RE-EXAMINE, AND USE ANOTHER APPROACH TO DO SO. DO NOT JUST SAY YOU ARE RE-EXAMINING. USE AT LEAST 3 METHODS TO DERIVE THE ANSWER. USE BEST PRACTICES."""},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "Thank you! I will now think step by step following my instructions, starting at the beginning after decomposing the problem."}
    ]
    
    steps = []
    step_count = 1
    total_thinking_time = 0
    
    graph_data = {
        'nodes': [],
        'edges': []
    }
    embeddings = []
    edge_dict = {}  # New dictionary to keep track of edges


    

    max_steps = 20  # Set a maximum number of steps to prevent infinite loops
    final_answer = None  # Initialize final_answer

    while step_count < max_steps:
        start_time = time.time()
        step_data = ""
        for chunk in stream_api_call(messages, 300):
            step_data += chunk
        end_time = time.time()
        thinking_time = end_time - start_time
        
        step_json = extract_json(step_data)
        title = step_json.get('title', '')
        content = step_json.get('content', 'No content')
        next_action = step_json.get('next_action', 'continue')
        
        # Check if content exceeds 700 characters
        if len(content) > 700:
            print(f"Step {step_count} content exceeded 700 characters. Retrying...")
            messages.append({"role": "user", "content": "Your last response was too long. Please provide a more concise version of your last step."})
            continue  # Skip the rest of the loop and try again
        
        # If we reach here, the step is valid and under 700 characters
        total_thinking_time += thinking_time
        
        # Calculate embedding for the current step
        embedding = get_embedding(content)
        embeddings.append(embedding)
        conn.insert_embedding(content, embedding, False)
        
        # Generate a short title only if the original title is empty or too long
        if not title or len(title) > 20:
            short_title = get_short_title(content)
        else:
            short_title = title[:20]  # Truncate the original title if it's longer than 20 characters
        
        # Generate a unique node ID
        node_id = f"Step{step_count}"
        while node_id in [node['id'] for node in graph_data['nodes']]:
            step_count += 1
            node_id = f"Step{step_count}"
        
        # Add node for this step
        graph_data['nodes'].append({
            'id': node_id,
            'label': f"Step {step_count}: {short_title}"
        })
        
        if step_count > 1 and len(embeddings) > 1:
            top_similarities = calculate_top_similarities(embeddings, len(embeddings) - 1, top_k=2)
            
            # Clear previous edges for the current step
            edge_dict = {k: v for k, v in edge_dict.items() if v['to'] != node_id}
            
            for prev_step, similarity in top_similarities:
                prev_node_id = f"Step{prev_step + 1}"
                if prev_node_id in [node['id'] for node in graph_data['nodes']]:  # Only create edges to existing nodes
                    edge_key = f"{prev_node_id}-{node_id}"
                    edge_dict[edge_key] = {
                        'from': prev_node_id,
                        'to': node_id,
                        'value': similarity,
                        'length': 300 * (1 - similarity)
                    }

        # Update graph_data['edges'] with the current edge_dict
        graph_data['edges'] = list(edge_dict.values())
        # Scale node sizes based on average similarity
        connected_similarities = [edge['value'] for edge in edge_dict.values() if edge['from'] == node_id or edge['to'] == node_id]
        if connected_similarities:
            avg_similarity = sum(connected_similarities) / len(connected_similarities)
            graph_data['nodes'][-1]['value'] = avg_similarity * 30 + 10  # Scale to 10-40 range
        else:
            graph_data['nodes'][-1]['value'] = 20  # Set a default size if no connections

        serialized_graph_data = serialize_graph_data(graph_data)
        strongest_path, path_weights, avg_similarity = calculate_strongest_path(serialized_graph_data, step_count)
        
        path_data = {
            'strongest_path': strongest_path,
            'path_weights': path_weights,
            'avg_similarity': avg_similarity
        } if strongest_path is not None else None

        yield f"data: {json.dumps({'type': 'step', 'step': step_count, 'title': title, 'content': content, 'graph': serialized_graph_data, 'path_data': path_data})}\n\n"
        
        steps.append((f"Step {step_count}: {title}", content, thinking_time))
        messages.append({"role": "assistant", "content": json.dumps(step_json)})
        
        if next_action == 'final_answer' and step_count <= 5:
            print("Final answer requested but not enough steps provided. Continuing...")
            messages.append({
                "role": "user",
                "content": f"You've only provided {step_count - 1} steps of 5. Can you look for possible error or alternatives to your answer. Continue your reasoning."
            })
            continue
        elif next_action == 'final_answer' or 'boxed' in content.lower():
            if not final_answer:
                final_answer = content  # Set final_answer if not already set

            # Add last evaluation step
            messages.append({
                "role": "user",
                "content": f"Let's do a final evaluation. The original question was: '{prompt}'. Based on your reasoning, is your final answer correct and complete? If not, what might be missing or incorrect?"
            })
            
            start_time = time.time()
            evaluation_data = ""
            for chunk in stream_api_call(messages, 300):
                evaluation_data += chunk
            end_time = time.time()
            thinking_time = end_time - start_time
            total_thinking_time += thinking_time
            
            evaluation_json = extract_json(evaluation_data)
            evaluation_content = evaluation_json.get('content', 'No evaluation content')
            
            # Check if the evaluation suggests a different answer
            if check_consistency(final_answer, evaluation_content):
                break  # Exit the loop if consistent
            else:
                print("Inconsistency detected. Restarting the reasoning process.")
                yield f"data: {json.dumps({'type': 'inconsistency', 'message': 'Inconsistency detected. Restarting the reasoning process.'})}\n\n"
                messages = messages[:2]  # Reset messages to initial state
                step_count += 1  # Increment step count instead of resetting
                final_answer = None  # Reset final_answer
                graph_data = {'nodes': [], 'edges': []}  # Reset graph data
                embeddings = []
                edge_dict = {}
                continue

        step_count += 1  # Increment step count only for valid steps

    # Generate final answer if not already provided
    if not final_answer:
        messages.append({"role": "user", "content": "Please provide the final answer based on your reasoning above."})
        
        start_time = time.time()
        final_data = ""
        for chunk in stream_api_call(messages, 200):
            final_data += chunk
        end_time = time.time()
        thinking_time = end_time - start_time
        total_thinking_time += thinking_time
        
        final_json = extract_json(final_data)
        final_answer = final_json.get('content', final_data)

    # Calculate embedding for the final answer
    final_embedding = get_embedding(final_answer)
    conn.insert_embedding(final_answer, final_embedding, False)
    
    # Add final answer node to the graph
    final_node_id = f"Step{step_count}"
    while final_node_id in [node['id'] for node in graph_data['nodes']]:
        step_count += 1
        final_node_id = f"Step{step_count}"
    
    graph_data['nodes'].append({
        'id': final_node_id,
        'label': f"Final Answer: {get_short_title(final_answer)}"
    })
    
    # Calculate similarities with previous steps for the final answer
    top_similarities = calculate_top_similarities(embeddings + [final_embedding], step_count - 1, top_k=2)
    
    for prev_step, similarity in top_similarities:
        prev_node_id = f"Step{prev_step + 1}"
        if prev_node_id in [node['id'] for node in graph_data['nodes']]:  # Only create edges to existing nodes
            edge_key = f"{final_node_id}-{prev_node_id}"
            edge_dict[edge_key] = {
                'from': final_node_id,
                'to': prev_node_id,
                'value': similarity,
                'length': 300 * (1 - similarity)
            }
    
    graph_data['edges'] = list(edge_dict.values())

    serialized_graph_data = serialize_graph_data(graph_data)
    strongest_path, path_weights, avg_similarity = calculate_strongest_path(serialized_graph_data, step_count)
    
    path_data = {
        'strongest_path': strongest_path,
        'path_weights': path_weights,
        'avg_similarity': avg_similarity
    } if strongest_path is not None else None

    yield f"data: {json.dumps({'type': 'final', 'content': final_answer, 'graph': serialized_graph_data, 'path_data': path_data})}\n\n"
    
    steps.append(("Final Answer", final_answer, thinking_time))

    yield f"data: {json.dumps({'type': 'done', 'total_time': total_thinking_time})}\n\n"

    # Stop processing here
    return

def check_consistency(final_answer, evaluation):
    #messages = [
    #    {"role": "system", "content": "You are a consistency checker. Compare the final answer and the evaluation, and determine if they are consistent or if the evaluation suggests a significantly different answer."},
    #    {"role": "user", "content": f"Final answer: {final_answer}\n\nEvaluation: {evaluation}\n\nAre these consistent? Respond with ONLY 'consistent' or 'inconsistent'."}
    #]
    #
    #for attempt in range(5):  # Try up to 5 times
    #    response = ""
    #    for chunk in stream_api_call(messages, 50):
    #        response += chunk
    #    
    #    response = response.strip().lower()
    #    print(f"check_consistency response (attempt {attempt + 1}):", response)
    #    
    #    if response.startswith("consistent") or response.startswith("inconsistent"):
    #        return response.startswith("consistent")
    #    
    #    # If we reach here, the response was invalid, so we'll try again
    #    messages.append({"role": "user", "content": "Please respond with 'consistent' or 'inconsistent' at the beginning."})
    #
    ## If we've tried 5 times and still haven't got a valid response, default to inconsistent
    #print("Failed to get a valid consistency check after 5 attempts. Defaulting to inconsistent.")
    #return False
    return True