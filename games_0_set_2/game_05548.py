import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import heapq
import os
import os
import pygame



class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: ↑↓←→ to move. Find the keys and reach the red exit door. "
        "The walls whisper and distort when the unseen monster is near."
    )

    game_description = (
        "A top-down survival horror game. Escape a procedurally generated haunted mansion "
        "by finding keys while an unseen monster hunts you. The closer it gets, the more "
        "the world distorts."
    )

    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (10, 10, 15)
    COLOR_WALL = (40, 40, 50)
    COLOR_FLOOR = (25, 25, 30)
    COLOR_PLAYER = (180, 220, 255)
    COLOR_KEY = (255, 255, 100)
    COLOR_EXIT = (200, 50, 50)
    COLOR_TEXT = (220, 220, 220)
    COLOR_DISTORTION = (255, 255, 255)

    # Screen and Grid
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 32, 20  # 20x20 grid cells
    TILE_SIZE = 20

    # Game Parameters
    MAX_STEPS = 1000
    NUM_KEYS = 3
    MONSTER_DISTORTION_RANGE = 15
    MONSTER_WHISPER_RANGE = 12

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Set headless mode for Pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 24)
        self.font_whisper = pygame.font.Font(None, 20)
        
        self.whisper_words = ["closer", "behind you", "get out", "run", "can't hide", "found you"]

        # All game state attributes are initialized in reset()
        self.player_pos = None
        self.monster_pos = None
        self.key_locations = None
        self.exit_pos = None
        self.walls = None
        self.floor = None
        self.steps = None
        self.score = None
        self.keys_collected = None
        self.game_over = None
        self.monster_move_debt = None
        self.whispers = None
        self.np_random = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self._generate_level()

        self.steps = 0
        self.score = 0.0
        self.keys_collected = 0
        self.game_over = False
        self.monster_move_debt = 0.0
        self.whispers = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            obs = self._get_observation()
            # On termination, return the last observation and a reward of 0
            return obs, 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1
        # shift_held = action[2] == 1
        
        self.steps += 1
        
        prev_player_pos = list(self.player_pos)
        reward = -0.1 # Cost of living

        # 1. Update Player Position
        self._move_player(movement)
        
        # 2. Update Monster Position
        self._move_monster()
        
        # 3. Check for events and calculate rewards
        # Key collection
        if tuple(self.player_pos) in self.key_locations:
            self.key_locations.remove(tuple(self.player_pos))
            self.keys_collected += 1
            reward += 50.0 # Large reward for finding a key

        # Proximity rewards
        if self.keys_collected < self.NUM_KEYS and self.key_locations:
            dist_before = self._get_dist_to_nearest_key(prev_player_pos)
            dist_after = self._get_dist_to_nearest_key(self.player_pos)
            if dist_after < dist_before:
                reward += 1.0 # Reward for moving towards a key
        elif self.keys_collected == self.NUM_KEYS:
            dist_before = self._manhattan_distance(prev_player_pos, self.exit_pos)
            dist_after = self._manhattan_distance(self.player_pos, self.exit_pos)
            if dist_after < dist_before:
                reward += 2.0 # Reward for moving towards the exit
        
        # 4. Check for termination conditions
        terminated = False
        truncated = False
        if self.player_pos == list(self.monster_pos):
            reward = -100.0
            terminated = True
            self.game_over = True
        elif tuple(self.player_pos) == self.exit_pos and self.keys_collected == self.NUM_KEYS:
            reward = 100.0
            terminated = True
            self.game_over = True
        elif self.steps >= self.MAX_STEPS:
            truncated = True
            self.game_over = True

        self.score += reward
        
        # 5. Update whispers based on new monster position
        self._update_whispers()
        
        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "keys_collected": self.keys_collected,
            "player_pos": self.player_pos,
            "monster_pos": self.monster_pos,
        }

    # --- Helper Methods ---

    def _generate_level(self):
        # 1. Initialize grid full of walls
        self.walls = set()
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                self.walls.add((x, y))
        
        # 2. Carve paths using randomized DFS
        self.floor = set()
        start_node = (self.np_random.integers(1, self.GRID_WIDTH // 2) * 2 - 1, self.np_random.integers(1, self.GRID_HEIGHT // 2) * 2 - 1)
        stack = [start_node]
        self.walls.remove(start_node)
        self.floor.add(start_node)
        
        while stack:
            cx, cy = stack[-1]
            neighbors = []
            for dx, dy in [(0, -2), (0, 2), (-2, 0), (2, 0)]:
                nx, ny = cx + dx, cy + dy
                if 0 < nx < self.GRID_WIDTH - 1 and 0 < ny < self.GRID_HEIGHT - 1 and (nx, ny) in self.walls:
                    neighbors.append((nx, ny))
            
            if neighbors:
                # FIX: Correctly call choice to select one element from the list
                nx, ny = self.np_random.choice(neighbors)
                # Carve path to neighbor
                self.walls.remove((nx, ny))
                self.floor.add((nx, ny))
                self.walls.remove(((cx + nx) // 2, (cy + ny) // 2))
                self.floor.add(((cx + nx) // 2, (cy + ny) // 2))
                stack.append((nx, ny))
            else:
                stack.pop()

        # 3. Place entities
        floor_list = list(self.floor)
        # FIX: Correctly call choice to select one element from the list
        self.player_pos = list(self.np_random.choice(floor_list))

        # Use BFS to find distances for smart placement
        distances = self._bfs(self.player_pos)
        
        # Place exit at the furthest point from the player
        max_dist = -1
        farthest_point = None
        for pos, dist in distances.items():
            if dist > max_dist:
                max_dist = dist
                farthest_point = pos
        self.exit_pos = farthest_point

        # Place monster near the exit
        monster_candidates = [p for p, d in distances.items() if d > max_dist * 0.8]
        # FIX: Correctly call choice to select one element from the list
        self.monster_pos = list(self.np_random.choice(monster_candidates))

        # Place keys at mid-distance points, avoiding player/exit
        key_candidates = [p for p, d in distances.items() if max_dist * 0.2 < d < max_dist * 0.8]
        key_indices = self.np_random.choice(len(key_candidates), size=min(self.NUM_KEYS, len(key_candidates)), replace=False)
        self.key_locations = {key_candidates[i] for i in key_indices}

    def _bfs(self, start_pos):
        q = [(tuple(start_pos), 0)]
        visited = {tuple(start_pos)}
        distances = {tuple(start_pos): 0}
        
        head = 0
        while head < len(q):
            (cx, cy), dist = q[head]
            head += 1
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = cx + dx, cy + dy
                if (nx, ny) in self.floor and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    distances[(nx, ny)] = dist + 1
                    q.append(((nx, ny), dist + 1))
        return distances

    def _move_player(self, movement):
        px, py = self.player_pos
        if movement == 1: # Up
            py -= 1
        elif movement == 2: # Down
            py += 1
        elif movement == 3: # Left
            px -= 1
        elif movement == 4: # Right
            px += 1
        
        if (px, py) in self.floor:
            self.player_pos = [px, py]

    def _move_monster(self):
        # Difficulty scaling: monster gets faster over time
        speed_bonus = (self.steps // 200) * 0.05
        self.monster_move_debt += 1.0 + speed_bonus

        while self.monster_move_debt >= 1.0:
            path = self._a_star(self.monster_pos, self.player_pos)
            if path and len(path) > 1:
                # Add randomness: 20% chance to move randomly
                if self.np_random.random() < 0.2:
                    mx, my = self.monster_pos
                    neighbors = []
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = mx + dx, my + dy
                        if (nx, ny) in self.floor:
                            neighbors.append((nx, ny))
                    if neighbors:
                        # FIX: Correctly call choice to select one element from the list
                        self.monster_pos = list(self.np_random.choice(neighbors))
                else:
                    self.monster_pos = list(path[1])
            self.monster_move_debt -= 1.0
            # If monster lands on player, stop moving this turn
            if self.monster_pos == self.player_pos:
                break

    def _a_star(self, start, end):
        start, end = tuple(start), tuple(end)
        open_set = [(0, start)]
        came_from = {}
        g_score = {pos: float('inf') for pos in self.floor}
        g_score[start] = 0
        f_score = {pos: float('inf') for pos in self.floor}
        f_score[start] = self._manhattan_distance(start, end)

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor not in self.floor:
                    continue
                
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._manhattan_distance(neighbor, end)
                    if neighbor not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_dist_to_nearest_key(self, pos):
        if not self.key_locations:
            return float('inf')
        return min(self._manhattan_distance(pos, k) for k in self.key_locations)
        
    def _update_whispers(self):
        # Fade out old whispers
        self.whispers = [w for w in self.whispers if w[2] < w[3]]
        for whisper in self.whispers:
            whisper[2] += 1 # age
            
        # Add new whispers if monster is close
        dist_to_monster = self._manhattan_distance(self.player_pos, self.monster_pos)
        if dist_to_monster < self.MONSTER_WHISPER_RANGE:
            # Chance to spawn increases with proximity
            if self.np_random.random() < (1.0 - dist_to_monster / self.MONSTER_WHISPER_RANGE) * 0.2:
                text = self.np_random.choice(self.whisper_words)
                pos = (self.np_random.integers(50, self.SCREEN_WIDTH - 50), self.np_random.integers(50, self.SCREEN_HEIGHT - 50))
                max_age = self.np_random.integers(60, 120)
                self.whispers.append([text, pos, 0, max_age])

    def _render_game(self):
        # Draw floor and walls
        for y in range(self.GRID_HEIGHT):
            for x in range(self.GRID_WIDTH):
                rect = (x * self.TILE_SIZE, y * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
                color = self.COLOR_FLOOR if (x,y) in self.floor else self.COLOR_WALL
                pygame.draw.rect(self.screen, color, rect)

        # Draw exit
        ex, ey = self.exit_pos
        exit_rect = pygame.Rect(ex * self.TILE_SIZE, ey * self.TILE_SIZE, self.TILE_SIZE, self.TILE_SIZE)
        pygame.draw.rect(self.screen, self.COLOR_EXIT, exit_rect)
        if self.keys_collected == self.NUM_KEYS:
            pygame.gfxdraw.rectangle(self.screen, exit_rect, (*self.COLOR_KEY, 150))


        # Draw keys
        for kx, ky in self.key_locations:
            center = (int((kx + 0.5) * self.TILE_SIZE), int((ky + 0.5) * self.TILE_SIZE))
            pygame.draw.circle(self.screen, self.COLOR_KEY, center, self.TILE_SIZE // 4)
            pygame.gfxdraw.aacircle(self.screen, center[0], center[1], self.TILE_SIZE // 4, self.COLOR_KEY)

        # Draw player
        px, py = self.player_pos
        player_center = (int((px + 0.5) * self.TILE_SIZE), int((py + 0.5) * self.TILE_SIZE))
        pygame.draw.circle(self.screen, self.COLOR_PLAYER, player_center, self.TILE_SIZE // 3)
        pygame.gfxdraw.aacircle(self.screen, player_center[0], player_center[1], self.TILE_SIZE // 3, self.COLOR_PLAYER)
        
        # Render monster proximity effects
        self._render_distortion()

    def _render_distortion(self):
        dist_to_monster = self._manhattan_distance(self.player_pos, self.monster_pos)
        intensity = max(0, 1 - dist_to_monster / self.MONSTER_DISTORTION_RANGE)

        if intensity > 0:
            # Screen shake
            shake_x = self.np_random.uniform(-1, 1) * intensity * 4
            shake_y = self.np_random.uniform(-1, 1) * intensity * 4
            temp_surface = self.screen.copy()
            self.screen.fill(self.COLOR_BG)
            self.screen.blit(temp_surface, (shake_x, shake_y))
            
            # White noise/static particles
            num_particles = int(intensity * 500)
            for _ in range(num_particles):
                pos = (self.np_random.integers(0, self.SCREEN_WIDTH), self.np_random.integers(0, self.SCREEN_HEIGHT))
                alpha = self.np_random.integers(10, 100)
                pygame.gfxdraw.pixel(self.screen, pos[0], pos[1], (*self.COLOR_DISTORTION, alpha))
        
        # Render whispers
        for text, pos, age, max_age in self.whispers:
            alpha = int(math.sin(age / max_age * math.pi) * 100 * intensity)
            if alpha > 0:
                text_surf = self.font_whisper.render(text, True, self.COLOR_TEXT)
                text_surf.set_alpha(alpha)
                self.screen.blit(text_surf, pos)


    def _render_ui(self):
        time_text = f"Time Left: {self.MAX_STEPS - self.steps}"
        time_surf = self.font_ui.render(time_text, True, self.COLOR_TEXT)
        self.screen.blit(time_surf, (10, 10))

        keys_text = f"Keys: {self.keys_collected} / {self.NUM_KEYS}"
        keys_surf = self.font_ui.render(keys_text, True, self.COLOR_TEXT)
        self.screen.blit(keys_surf, (10, 35))

    def close(self):
        pygame.quit()
        
    def render(self):
        return self._get_observation()

if __name__ == "__main__":
    # Ensure Pygame runs headlessly
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    env = GameEnv()
    obs, info = env.reset(seed=42)
    
    # Test a few steps
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"Step {i+1}: Action: {action}, Reward: {reward:.2f}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")
        if terminated or truncated:
            print("Game Over.")
            break
            
    print(f"\nTest finished. Final score: {info['score']}")
    
    # --- Example of how to run with visualization ---
    # To see the game, you would need a display. 
    # 1. Comment out the `os.environ["SDL_VIDEODRIVER"] = "dummy"` line in __init__ and here.
    # 2. Add a "human" render mode and handle display updates.
    
    # try:
    #     # This part will fail in a headless environment but shows how to run with a display.
    #     del os.environ["SDL_VIDEODRIVER"]
    #     import pygame
    #     pygame.display.init()
    #     pygame.font.init()
        
    #     env = GameEnv(render_mode="human")
    #     screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    #     pygame.display.set_caption("Haunted Mansion")
    #     clock = pygame.time.Clock()
        
    #     obs, info = env.reset()
    #     done = False
    #     while not done:
    #         # Simple agent: move randomly
    #         action = env.action_space.sample()
            
    #         obs, reward, terminated, truncated, info = env.step(action)
    #         done = terminated or truncated
            
    #         # Render the observation to the screen
    #         frame = np.transpose(obs, (1, 0, 2))
    #         surf = pygame.surfarray.make_surface(frame)
    #         screen.blit(surf, (0, 0))
    #         pygame.display.flip()
            
    #         # Handle quit event
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 done = True
            
    #         clock.tick(10) # Limit frame rate
            
    #     print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
    
    # except pygame.error as e:
    #     print("\nSkipping visualization example (no display available).")

    env.close()