import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T17:32:07.712448
# Source Brief: brief_02905.md
# Brief Index: 2905
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
from collections import deque

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Navigate a shifting isometric maze to collect glowing orbs. "
        "Grab special red orbs to alter the maze's layout, but be quick before time runs out!"
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. "
        "Press space when on a tile with an orb to collect it."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    MAZE_WIDTH, MAZE_HEIGHT = 15, 15
    TILE_WIDTH, TILE_HEIGHT = 48, 24
    WALL_HEIGHT = 32
    MAX_STEPS = 1200
    WIN_SCORE = 10
    ORB_LIFETIME = 50
    ORB_SPAWN_CHANCE = 0.1
    MAX_ORBS = 5
    RED_ORB_CHANCE = 0.2

    # --- Colors ---
    COLOR_BG = (15, 18, 32)
    COLOR_FLOOR = (40, 45, 60)
    COLOR_FLOOR_ACCENT = (50, 55, 70)
    COLOR_WALL_TOP = (80, 85, 100)
    COLOR_WALL_SIDE = (65, 70, 85)
    COLOR_PLAYER = (255, 255, 255)
    COLOR_PLAYER_GLOW = (200, 200, 255)
    COLOR_ORB_YELLOW = (255, 220, 0)
    COLOR_ORB_YELLOW_GLOW = (180, 150, 0)
    COLOR_ORB_RED = (255, 50, 50)
    COLOR_ORB_RED_GLOW = (180, 40, 40)
    COLOR_UI_TEXT = (220, 220, 240)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 60)

        self.render_mode = render_mode
        self.origin_x = self.SCREEN_WIDTH // 2
        self.origin_y = self.SCREEN_HEIGHT // 2 - (self.MAZE_HEIGHT * self.TILE_HEIGHT) // 4
        
        # State variables will be initialized in reset()
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.maze = np.zeros((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.int8)
        self.player_pos = [0, 0]
        self.player_visual_pos = [0.0, 0.0]
        self.orbs = []
        self.particles = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        self._generate_maze()
        
        # Find a valid starting position
        start_pos = self._find_random_empty_cell()
        self.player_pos = list(start_pos)
        self.player_visual_pos = list(start_pos)
        
        self.orbs = []
        self.particles = []
        
        for _ in range(self.MAX_ORBS // 2):
            self._spawn_orb()

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, _ = action[0], action[1] == 1, action[2] == 1
        reward = 0
        
        # --- Pre-computation for reward ---
        dist_before, _ = self._find_nearest_orb()

        # --- Handle Movement ---
        target_pos = self.player_pos[:]
        if movement == 1: target_pos[1] -= 1 # Up (North)
        elif movement == 2: target_pos[1] += 1 # Down (South)
        elif movement == 3: target_pos[0] -= 1 # Left (West)
        elif movement == 4: target_pos[0] += 1 # Right (East)
        
        if self._is_valid_pos(target_pos[0], target_pos[1]):
            self.player_pos = target_pos

        # --- Handle Orb Collection ---
        if space_held:
            collected_orb = None
            for orb in self.orbs:
                if self.player_pos[0] == orb['pos'][0] and self.player_pos[1] == orb['pos'][1]:
                    collected_orb = orb
                    break
            
            if collected_orb:
                # Sfx: Orb collect sound
                self.score += 1
                self.orbs.remove(collected_orb)
                self._create_particles(collected_orb['pos'], collected_orb['color'])

                if collected_orb['type'] == 'red':
                    reward += 15
                    self._shift_wall()
                else:
                    reward += 10

        # --- Update Game State ---
        self.steps += 1
        self._update_orbs()
        self._update_particles()
        if self.np_random.random() < self.ORB_SPAWN_CHANCE and len(self.orbs) < self.MAX_ORBS:
            self._spawn_orb()
        
        # Ensure at least one orb exists if not won
        if not self.orbs and self.score < self.WIN_SCORE:
            self._spawn_orb()

        # --- Calculate Proximity Reward ---
        dist_after, _ = self._find_nearest_orb()
        if dist_before is not None and dist_after is not None:
            if dist_after < dist_before:
                reward += 0.1

        # --- Check Termination ---
        terminated = (self.score >= self.WIN_SCORE) or (self.steps >= self.MAX_STEPS)
        truncated = False # This environment does not truncate based on conditions other than termination
        if terminated and not self.game_over:
            self.game_over = True
            if self.score >= self.WIN_SCORE:
                reward += 100 # Win bonus
            else:
                reward -= 100 # Timeout penalty

        obs = self._get_observation()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps}

    # --- Rendering ---
    def _project(self, x, y, z=0):
        """Projects 3D maze coordinates to 2D screen coordinates."""
        sx = self.origin_x + (x - y) * self.TILE_WIDTH / 2
        sy = self.origin_y + (x + y) * self.TILE_HEIGHT / 2 - z
        return int(sx), int(sy)

    def _render_game(self):
        # Interpolate player visual position for smooth movement
        interp_rate = 0.25
        self.player_visual_pos[0] += (self.player_pos[0] - self.player_visual_pos[0]) * interp_rate
        self.player_visual_pos[1] += (self.player_pos[1] - self.player_visual_pos[1]) * interp_rate

        # Draw from back to front
        for y in range(self.MAZE_HEIGHT):
            for x in range(self.MAZE_WIDTH):
                is_wall = self.maze[y, x] == 1
                
                # Draw floor tile
                if not is_wall:
                    floor_points = [
                        self._project(x, y), self._project(x + 1, y),
                        self._project(x + 1, y + 1), self._project(x, y + 1)
                    ]
                    color = self.COLOR_FLOOR_ACCENT if (x + y) % 2 == 0 else self.COLOR_FLOOR
                    pygame.draw.polygon(self.screen, color, floor_points)

                # Draw elements on this tile (orbs, player)
                # This ensures correct occlusion
                for orb in self.orbs:
                    if orb['pos'][0] == x and orb['pos'][1] == y:
                        self._draw_orb(orb)
                
                if self.player_pos[0] == x and self.player_pos[1] == y:
                     self._draw_player()
                
                # Draw wall
                if is_wall:
                    self._draw_wall(x, y)
        
        # Draw particles on top of everything
        for p in self.particles:
            pos = self._project(p['pos'][0], p['pos'][1], p['pos'][2])
            pygame.draw.circle(self.screen, p['color'], pos, int(p['size']))

    def _draw_wall(self, x, y):
        # Top face
        top_points = [
            self._project(x, y, self.WALL_HEIGHT), self._project(x + 1, y, self.WALL_HEIGHT),
            self._project(x + 1, y + 1, self.WALL_HEIGHT), self._project(x, y + 1, self.WALL_HEIGHT)
        ]
        pygame.draw.polygon(self.screen, self.COLOR_WALL_TOP, top_points)
        
        # South face
        if y + 1 >= self.MAZE_HEIGHT or self.maze[y + 1, x] == 0:
            face_points = [
                self._project(x, y + 1), self._project(x + 1, y + 1),
                self._project(x + 1, y + 1, self.WALL_HEIGHT), self._project(x, y + 1, self.WALL_HEIGHT)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_WALL_SIDE, face_points)
        
        # East face
        if x + 1 >= self.MAZE_WIDTH or self.maze[y, x + 1] == 0:
            face_points = [
                self._project(x + 1, y), self._project(x + 1, y + 1),
                self._project(x + 1, y + 1, self.WALL_HEIGHT), self._project(x + 1, y, self.WALL_HEIGHT)
            ]
            pygame.draw.polygon(self.screen, self.COLOR_WALL_SIDE, face_points)

    def _draw_orb(self, orb):
        pulse = (math.sin(self.steps * 0.1 + orb['pos'][0]) + 1) / 2 # 0 to 1
        z_offset = 10 + pulse * 5
        base_radius = 8
        
        pos = self._project(orb['pos'][0] + 0.5, orb['pos'][1] + 0.5, z_offset)
        
        # Glow
        glow_radius = int(base_radius * (1.5 + pulse * 0.5))
        glow_color = (*orb['glow_color'], 80) # Use alpha for glow
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, glow_color, (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))

        # Core
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], base_radius, orb['color'])
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], base_radius, orb['color'])

    def _draw_player(self):
        z_offset = 12
        px, py = self.player_visual_pos
        pos = self._project(px + 0.5, py + 0.5, z_offset)
        
        # Glow
        glow_radius = 20
        s = pygame.Surface((glow_radius*2, glow_radius*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.COLOR_PLAYER_GLOW, 60), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(s, (pos[0] - glow_radius, pos[1] - glow_radius))

        # Core
        radius = 9
        pygame.gfxdraw.filled_circle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, pos[0], pos[1], radius, self.COLOR_PLAYER)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Orbs: {self.score} / {self.WIN_SCORE}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (10, 10))
        
        # Timer
        time_left = self.MAX_STEPS - self.steps
        timer_text = self.font_ui.render(f"Time: {time_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(timer_text, (self.SCREEN_WIDTH - timer_text.get_width() - 10, 10))

        # Game Over Message
        if self.game_over:
            msg = "YOU WIN!" if self.score >= self.WIN_SCORE else "TIME OUT"
            color = self.COLOR_ORB_YELLOW if self.score >= self.WIN_SCORE else self.COLOR_ORB_RED
            end_text = self.font_big.render(msg, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    # --- Game Logic Helpers ---
    def _generate_maze(self):
        self.maze = np.ones((self.MAZE_HEIGHT, self.MAZE_WIDTH), dtype=np.int8)
        
        # Use randomized Prim's algorithm
        start_x, start_y = (self.np_random.integers(1, self.MAZE_WIDTH-1, endpoint=False)//2)*2+1, \
                           (self.np_random.integers(1, self.MAZE_HEIGHT-1, endpoint=False)//2)*2+1
        
        self.maze[start_y, start_x] = 0
        
        frontiers = []
        for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            nx, ny = start_x + dx, start_y + dy
            if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT:
                frontiers.append((nx, ny))

        while frontiers:
            fx, fy = frontiers.pop(self.np_random.integers(len(frontiers)))
            
            neighbors = []
            for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                nx, ny = fx - dx, fy - dy
                if 0 <= nx < self.MAZE_WIDTH and 0 <= ny < self.MAZE_HEIGHT and self.maze[ny, nx] == 0:
                    neighbors.append((nx, ny))
            
            if neighbors:
                nx, ny = neighbors[self.np_random.integers(len(neighbors))]
                self.maze[fy, fx] = 0
                self.maze[fy - (fy-ny)//2, fx - (fx-nx)//2] = 0

                for dx, dy in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
                    nfx, nfy = fx + dx, fy + dy
                    if 0 <= nfx < self.MAZE_WIDTH and 0 <= nfy < self.MAZE_HEIGHT and self.maze[nfy, nfx] == 1:
                        if (nfx, nfy) not in frontiers:
                            frontiers.append((nfx, nfy))

    def _is_valid_pos(self, x, y):
        return 0 <= x < self.MAZE_WIDTH and 0 <= y < self.MAZE_HEIGHT and self.maze[y, x] == 0

    def _find_random_empty_cell(self):
        while True:
            x = self.np_random.integers(self.MAZE_WIDTH)
            y = self.np_random.integers(self.MAZE_HEIGHT)
            if self._is_valid_pos(x, y):
                return x, y

    def _spawn_orb(self):
        pos = self._find_random_empty_cell()
        is_red = self.np_random.random() < self.RED_ORB_CHANCE
        orb_type = 'red' if is_red else 'yellow'
        self.orbs.append({
            'pos': list(pos),
            'type': orb_type,
            'color': self.COLOR_ORB_RED if is_red else self.COLOR_ORB_YELLOW,
            'glow_color': self.COLOR_ORB_RED_GLOW if is_red else self.COLOR_ORB_YELLOW_GLOW,
            'lifetime': self.ORB_LIFETIME
        })

    def _update_orbs(self):
        for orb in self.orbs[:]:
            orb['lifetime'] -= 1
            if orb['lifetime'] <= 0:
                self.orbs.remove(orb)

    def _shift_wall(self):
        # Find a wall that can be removed and a path that can be filled
        removable_walls = []
        fillable_paths = []
        for y in range(1, self.MAZE_HEIGHT - 1):
            for x in range(1, self.MAZE_WIDTH - 1):
                # A removable wall is surrounded by paths in a line (e.g., path-wall-path)
                if self.maze[y, x] == 1:
                    if self.maze[y-1, x] == 0 and self.maze[y+1, x] == 0:
                        removable_walls.append((x, y))
                    elif self.maze[y, x-1] == 0 and self.maze[y, x+1] == 0:
                        removable_walls.append((x, y))
                # A fillable path is a dead end
                elif self.maze[y, x] == 0:
                    wall_neighbors = sum([self.maze[y-1, x], self.maze[y+1, x], self.maze[y, x-1], self.maze[y, x+1]])
                    if wall_neighbors >= 3:
                        fillable_paths.append((x, y))
        
        if removable_walls and fillable_paths:
            # Sfx: Wall shifting sound
            wall_to_remove = removable_walls[self.np_random.integers(len(removable_walls))]
            path_to_fill = fillable_paths[self.np_random.integers(len(fillable_paths))]
            
            # Ensure we don't pick the same spot, although unlikely
            if wall_to_remove != path_to_fill:
                self.maze[wall_to_remove[1], wall_to_remove[0]] = 0 # Open path
                self.maze[path_to_fill[1], path_to_fill[0]] = 1 # Close path

    def _find_nearest_orb(self):
        if not self.orbs:
            return None, None
        
        min_dist = float('inf')
        nearest_orb = None
        for orb in self.orbs:
            dist = abs(self.player_pos[0] - orb['pos'][0]) + abs(self.player_pos[1] - orb['pos'][1])
            if dist < min_dist:
                min_dist = dist
                nearest_orb = orb
        return min_dist, nearest_orb

    def _create_particles(self, pos, color):
        # Sfx: Particle burst sound
        for _ in range(20):
            self.particles.append({
                'pos': [pos[0] + 0.5, pos[1] + 0.5, self.WALL_HEIGHT / 2],
                'vel': [(self.np_random.random() - 0.5) * 0.3, (self.np_random.random() - 0.5) * 0.3, (self.np_random.random()) * 0.5],
                'size': self.np_random.random() * 3 + 1,
                'lifetime': 20,
                'color': color
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['pos'][2] += p['vel'][2]
            p['vel'][2] -= 0.05 # Gravity
            p['lifetime'] -= 1
            p['size'] *= 0.95
            if p['lifetime'] <= 0:
                self.particles.remove(p)

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    # To run this, you need to unset the dummy video driver
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    pygame.display.set_caption("3D Maze Orb Collector")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    # --- Manual Play Loop ---
    while running:
        movement = 0 # no-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP] or keys[pygame.K_w]: movement = 1
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]: movement = 2
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]: movement = 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1
        
        action = [movement, space_held, shift_held]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Display the observation from the environment
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated or truncated:
            print(f"Episode finished. Total Reward: {total_reward}, Score: {info['score']}, Steps: {info['steps']}")
            total_reward = 0
            obs, info = env.reset()
            pygame.time.wait(2000) # Pause before restarting

        clock.tick(30) # Limit frame rate for manual play
        
    env.close()