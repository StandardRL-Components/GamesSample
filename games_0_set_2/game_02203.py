
# Generated: 2025-08-28T04:04:05.750836
# Source Brief: brief_02203.md
# Brief Index: 2203

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
import os
import pygame
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move. Avoid the red monsters and reach the green exit before time runs out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate a dark, procedurally generated maze. Evade patrolling monsters and race against the clock to find the exit. Each step brings you closer to victory or doom."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.CELL_SIZE = 20
        self.GRID_WIDTH = self.WIDTH // self.CELL_SIZE
        self.GRID_HEIGHT = self.HEIGHT // self.CELL_SIZE
        self.MAX_STEPS = 1200  # 120 seconds at 10 steps/sec
        self.NUM_MONSTERS = 3

        # Colors
        self.COLOR_BG = (20, 20, 30)
        self.COLOR_WALL = (60, 60, 80)
        self.COLOR_PLAYER = (255, 200, 0)
        self.COLOR_PLAYER_GLOW = (255, 200, 0, 50)
        self.COLOR_EXIT = (0, 255, 120)
        self.COLOR_MONSTER = (255, 50, 50)
        self.COLOR_TEXT = (240, 240, 240)
        
        # Gymnasium spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 50, bold=True)
        
        # State variables (initialized in reset)
        self.maze = None
        self.player_pos = None
        self.exit_pos = None
        self.monsters = []
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.last_dist_to_exit = 0
        self.monster_speed = 0
        self.monster_move_debt = 0

        # Run validation
        # self.validate_implementation() # Commented out for submission, but useful for testing

    def _generate_maze(self):
        # Maze generation using recursive backtracking
        w, h = (self.GRID_WIDTH - 1) // 2, (self.GRID_HEIGHT - 1) // 2
        maze = np.ones((self.GRID_HEIGHT, self.GRID_WIDTH), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)
        
        def carve(x, y):
            visited[y, x] = True
            maze[y*2+1, x*2+1] = 0
            
            directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            self.np_random.shuffle(directions)
            
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                    maze[y*2+1 + dy, x*2+1 + dx] = 0
                    carve(nx, ny)

        carve(self.np_random.integers(w), self.np_random.integers(h))
        return maze

    def _get_valid_spawn_points(self, count):
        corridors = np.argwhere(self.maze == 0)
        if len(corridors) < count:
            raise ValueError("Not enough empty cells in the maze to spawn entities.")
        indices = self.np_random.choice(len(corridors), size=count, replace=False)
        return [tuple(p) for p in corridors[indices]]

    def _find_path(self, start, end):
        # Simple Breadth-First Search for pathfinding
        q = [(start, [start])]
        visited = {start}
        while q:
            (cy, cx), path = q.pop(0)
            if (cy, cx) == end:
                return path
            for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < self.GRID_HEIGHT and 0 <= nx < self.GRID_WIDTH and self.maze[ny, nx] == 0 and (ny, nx) not in visited:
                    visited.add((ny, nx))
                    q.append(((ny, nx), path + [(ny, nx)]))
        return []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Generate maze and spawn points
        self.maze = self._generate_maze()
        
        # Ensure player and exit are far apart
        while True:
            spawns = self._get_valid_spawn_points(2)
            p_pos, e_pos = spawns[0], spawns[1]
            dist = abs(p_pos[0] - e_pos[0]) + abs(p_pos[1] - e_pos[1])
            if dist > (self.GRID_WIDTH + self.GRID_HEIGHT) / 3:
                self.player_pos = list(p_pos)
                self.exit_pos = e_pos
                break
        
        self.last_dist_to_exit = abs(self.player_pos[0] - self.exit_pos[0]) + abs(self.player_pos[1] - self.exit_pos[1])

        # Initialize monsters
        self.monsters = []
        monster_spawns = self._get_valid_spawn_points(self.NUM_MONSTERS)
        for pos in monster_spawns:
            waypoints = [pos] + self._get_valid_spawn_points(self.np_random.integers(3, 6))
            monster = {
                'pos': list(pos),
                'waypoints': waypoints,
                'waypoint_idx': 1,
                'path': self._find_path(tuple(pos), tuple(waypoints[1])),
                'path_idx': 0,
                'dir': (0, 0)
            }
            self.monsters.append(monster)

        # Reset state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.particles.clear()
        self.monster_speed = 1.0
        self.monster_move_debt = 0.0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]
        # space_held and shift_held are ignored per the brief
        
        reward = -0.01  # Small penalty for each step
        
        # --- Player Movement ---
        py, px = self.player_pos
        ny, nx = py, px
        if movement == 1: ny -= 1  # Up
        elif movement == 2: ny += 1  # Down
        elif movement == 3: nx -= 1  # Left
        elif movement == 4: nx += 1  # Right
        
        if self.maze[ny, nx] == 0:
            self.player_pos = [ny, nx]

        # --- Reward for distance change ---
        current_dist = abs(self.player_pos[0] - self.exit_pos[0]) + abs(self.player_pos[1] - self.exit_pos[1])
        if current_dist < self.last_dist_to_exit:
            reward += 0.1
        elif current_dist > self.last_dist_to_exit:
            reward -= 0.1
        self.last_dist_to_exit = current_dist

        # --- Monster AI and Movement ---
        if self.steps > 0 and self.steps % 200 == 0:
            self.monster_speed += 0.05
        
        self.monster_move_debt += self.monster_speed
        moves_to_make = int(self.monster_move_debt)
        
        for _ in range(moves_to_make):
            for m in self.monsters:
                if not m['path']:
                    continue
                
                # Create particle at old position
                # SFX: Monster step
                self.particles.append({
                    'pos': [m['pos'][1] * self.CELL_SIZE + self.CELL_SIZE / 2, m['pos'][0] * self.CELL_SIZE + self.CELL_SIZE / 2],
                    'life': 10, 'max_life': 10, 'size': 4, 'color': self.COLOR_MONSTER
                })
                
                old_pos = m['pos']
                m['path_idx'] += 1
                if m['path_idx'] >= len(m['path']):
                    # Reached waypoint, get next one
                    m['waypoint_idx'] = (m['waypoint_idx'] + 1) % len(m['waypoints'])
                    next_waypoint = tuple(m['waypoints'][m['waypoint_idx']])
                    m['path'] = self._find_path(tuple(m['pos']), next_waypoint)
                    m['path_idx'] = 0 if not m['path'] else 1

                if m['path'] and m['path_idx'] < len(m['path']):
                    m['pos'] = list(m['path'][m['path_idx']])
                    m['dir'] = (m['pos'][0] - old_pos[0], m['pos'][1] - old_pos[1])
        
        self.monster_move_debt -= moves_to_make

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['life'] -= 1

        # --- Update state ---
        self.steps += 1
        self.score += reward
        
        # --- Check Termination ---
        terminated = False
        # 1. Collision with monster
        for m in self.monsters:
            if self.player_pos == m['pos']:
                self.game_over = True
                terminated = True
                # SFX: Player death
                break
        
        # 2. Reached exit
        if not terminated and tuple(self.player_pos) == self.exit_pos:
            self.game_over = True
            self.win = True
            terminated = True
            reward += 100
            self.score += 100
            # SFX: Level complete
        
        # 3. Timeout
        if not terminated and self.steps >= self.MAX_STEPS:
            self.game_over = True
            terminated = True
            # SFX: Timeout
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _render_game(self):
        # Render maze walls
        for r in range(self.GRID_HEIGHT):
            for c in range(self.GRID_WIDTH):
                if self.maze[r, c] == 1:
                    pygame.draw.rect(self.screen, self.COLOR_WALL, (c * self.CELL_SIZE, r * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Render exit
        ex, ey = self.exit_pos[1], self.exit_pos[0]
        pygame.draw.rect(self.screen, self.COLOR_EXIT, (ex * self.CELL_SIZE, ey * self.CELL_SIZE, self.CELL_SIZE, self.CELL_SIZE))

        # Render particles
        for p in self.particles:
            alpha = int(255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            s = pygame.Surface((p['size'], p['size']), pygame.SRCALPHA)
            pygame.draw.rect(s, color, s.get_rect())
            self.screen.blit(s, (p['pos'][0] - p['size']//2, p['pos'][1] - p['size']//2))

        # Render monsters
        for m in self.monsters:
            my, mx = m['pos']
            center_x, center_y = mx * self.CELL_SIZE + self.CELL_SIZE / 2, my * self.CELL_SIZE + self.CELL_SIZE / 2
            
            # Simple rotation based on direction
            dy, dx = m['dir']
            angle = math.atan2(-dy, dx) # atan2(y,x)
            
            s = self.CELL_SIZE * 0.4
            p1 = (center_x + s * math.cos(angle), center_y + s * math.sin(angle))
            p2 = (center_x + s * math.cos(angle + 2.356), center_y + s * math.sin(angle + 2.356)) # 135 deg
            p3 = (center_x + s * math.cos(angle - 2.356), center_y + s * math.sin(angle - 2.356)) # -135 deg
            
            points = [p1, p2, p3]
            int_points = [(int(p[0]), int(p[1])) for p in points]
            
            pygame.gfxdraw.aapolygon(self.screen, int_points, self.COLOR_MONSTER)
            pygame.gfxdraw.filled_polygon(self.screen, int_points, self.COLOR_MONSTER)

        # Render player
        py, px = self.player_pos
        player_center_x = px * self.CELL_SIZE + self.CELL_SIZE // 2
        player_center_y = py * self.CELL_SIZE + self.CELL_SIZE // 2
        
        # Glow effect
        glow_radius = int(self.CELL_SIZE * 0.8)
        s = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(s, self.COLOR_PLAYER_GLOW, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (player_center_x - glow_radius, player_center_y - glow_radius))
        
        # Player square
        player_rect = pygame.Rect(
            px * self.CELL_SIZE + self.CELL_SIZE * 0.2, 
            py * self.CELL_SIZE + self.CELL_SIZE * 0.2, 
            self.CELL_SIZE * 0.6, 
            self.CELL_SIZE * 0.6
        )
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer (remaining steps)
        time_left = max(0, self.MAX_STEPS - self.steps)
        time_text = self.font_ui.render(f"TIME: {time_left}", True, self.COLOR_TEXT)
        self.screen.blit(time_text, (self.WIDTH - time_text.get_width() - 10, 10))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            msg = "YOU ESCAPED!" if self.win else "YOU WERE CAUGHT"
            color = self.COLOR_EXIT if self.win else self.COLOR_MONSTER
            
            end_text = self.font_game_over.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH / 2, self.HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

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
            "player_pos": self.player_pos,
            "exit_pos": self.exit_pos,
            "dist_to_exit": self.last_dist_to_exit
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        print("Running implementation validation...")
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        # Reset must be called before get_observation
        self.reset()
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3), f"Obs shape is {test_obs.shape}"
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game manually for testing
    env = GameEnv()
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Maze Runner")
    clock = pygame.time.Clock()
    
    done = False
    total_reward = 0
    
    # Game loop
    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Render the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print(f"Game Over! Final Score: {info['score']:.2f}, Steps: {info['steps']}")
            pygame.time.wait(2000) # Pause before closing
            break

        # Since auto_advance is False, we need to control the speed of human play
        clock.tick(10) # Limit to 10 actions per second for human playability

    env.close()