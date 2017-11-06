import queue

graph = {'A': set(['B', 'C']),
         'B': set(['A', 'D', 'E']),
         'C': set(['A', 'F']),
         'D': set(['B']),
         'E': set(['B', 'F']),
         'F': set(['C', 'E'])}

def bfs(graph,root):
	q = queue.Queue()
	visited = []
	q.put(root)
	while( not q.empty() ):
		current = q.get()
		if current not in visited:
			visited.append(current)
			for node in graph[current]:
				q.put(node)

	return visited

print("bfs",bfs(graph,'A'))


def interative_DFS(graph,start):
	stack = [start]
	visited = list()
	while (stack):
		current = stack.pop()
		if current not in visited:
			visited.append(current)
			stack += graph[current]
	return visited

path = interative_DFS(graph,'A')
print("i_dfs",path)


def recursive_dfs(graph,start,visited):
	visited.append(start)
	for node in graph[start]:
		if node not in visited:
			visited = recursive_dfs(graph,node,visited)
	return visited

print("r_dfs",recursive_dfs(graph,'A',[]))

			



