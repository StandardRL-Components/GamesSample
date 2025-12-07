
# Generated: 2025-08-27T15:39:14.789683
# Source Brief: brief_01034.md
# Brief Index: 1034

        
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
        "Controls: Use arrow keys to select a tile. Press space to move the nearest crystal to the selected tile."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Move crystals to align 5 in a row or column before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((640, 400))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("monospace", 16)
        self.font_large = pygame.font.SysFont("monospace", 32, bold=True)
        
        # Game constants
        self.GRID_WIDTH = 10
        self.GRID_HEIGHT = 10
        self.TILE_WIDTH_HALF = 24
        self.TILE_HEIGHT_HALF = 12
        self.MAX_MOVES = 20
        self.NUM_CRYSTALS = 5
        self.WIN_ALIGNMENT = 5
        self.WORLD_ORIGIN = (self.screen.get_width() // 2, 80)

        # Colors
        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_WALL = (60, 70, 90)
        self.COLOR_WALL_TOP = (80, 90, 110)
        self.CRYSTAL_COLORS = [
            (100, 255, 255),  # Cyan
            (255, 100, 255),  # Magenta
            (255, 255, 100),  # Yellow
            (100, 255, 100),  # Green
            (255, 150, 100),  # Orange
        ]
        self.COLOR_CURSOR = (200, 220, 255)
        self.COLOR_UI_TEXT = (220, 220, 220)
        
        # State variables (will be initialized in reset)
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = 0
        self.cursor_pos = (0, 0)
        self.crystals = []
        self.walls = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        self.particles = []
        self.last_space_press = False
        self.last_alignment_pairs = 0
        self.max_align = 0
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_left = self.MAX_MOVES
        
        self._generate_level()
        
        self.cursor_pos = (self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2)
        self.particles = []
        self.last_space_press = False
        
        self.max_align, self.last_alignment_pairs = self._calculate_alignment()
        
        # MUST return exactly this tuple
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        space_held = action[1] == 1  # Boolean
        shift_held = action[2] == 1  # Boolean (unused)
        
        reward = 0
        terminated = False
        
        # --- Handle Cursor Movement ---
        cx, cy = self.cursor_pos
        if movement == 1: cy -= 1  # Up
        elif movement == 2: cy += 1  # Down
        elif movement == 3: cx -= 1  # Left
        elif movement == 4: cx += 1  # Right
        
        self.cursor_pos = (
            np.clip(cx, 0, self.GRID_WIDTH - 1),
            np.clip(cy, 0, self.GRID_HEIGHT - 1)
        )
        
        # --- Handle Move Action ---
        move_executed = False
        if space_held and not self.last_space_press:
            move_executed = self._execute_move()

        if move_executed:
            self.moves_left -= 1
            self.steps += 1
            
            # Update alignment and calculate reward
            new_max_align, new_alignment_pairs = self._calculate_alignment()
            
            # Continuous feedback reward for creating pairs
            reward += (new_alignment_pairs - self.last_alignment_pairs) * 0.1
            
            # Event-based reward for increasing the max alignment chain
            if new_max_align > self.max_align:
                reward += (new_max_align - self.max_align) * 1.0

            self.score += reward
            self.max_align = new_max_align
            self.last_alignment_pairs = new_alignment_pairs
            
            # Check for termination
            if self.max_align >= self.WIN_ALIGNMENT:
                reward += 100
                self.score += 100
                terminated = True
                self.game_over = True
            elif self.moves_left <= 0:
                reward -= 100
                self.score -= 100
                terminated = True
                self.game_over = True
        
        self.last_space_press = space_held
        
        self._update_particles()

        # MUST return exactly this 5-tuple
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )
    
    def _get_observation(self):
        # Clear screen with background
        self.screen.fill(self.COLOR_BG)
        
        # Render all game elements
        self._render_game()
        
        # Render UI overlay
        self._render_ui()
        
        # Convert to numpy array (EXACT format required)
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)
    
    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "max_alignment": self.max_align,
        }

    def _generate_level(self):
        self.walls = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        
        # Create a simple border wall, leaving the top open
        for y in range(self.GRID_HEIGHT):
            self.walls[0][y] = 1
            self.walls[self.GRID_WIDTH - 1][y] = 1
        for x in range(self.GRID_WIDTH):
            self.walls[x][self.GRID_HEIGHT - 1] = 1

        # Add some random pillars
        for _ in range(self.np_random.integers(3, 6)):
            px, py = self.np_random.integers(1, self.GRID_WIDTH - 1), self.np_random.integers(1, self.GRID_HEIGHT - 2)
            self.walls[px][py] = 1
            
        # Place crystals
        self.crystals = []
        occupied_coords = set()
        while len(self.crystals) < self.NUM_CRYSTALS:
            cx, cy = self.np_random.integers(1, self.GRID_WIDTH - 1), self.np_random.integers(0, self.GRID_HEIGHT - 2)
            if self.walls[cx][cy] == 0 and (cx, cy) not in occupied_coords:
                self.crystals.append((cx, cy))
                occupied_coords.add((cx, cy))
        
        # Settle crystals initially
        self._apply_gravity(spawn_particles=False)

    def _execute_move(self):
        target_x, target_y = self.cursor_pos
        
        # Check if target is valid (not a wall or occupied by another crystal)
        if self.walls[target_x][target_y] == 1 or (target_x, target_y) in self.crystals:
            # sfx: invalid move sound
            return False

        if not self.crystals: return False
        
        # Find the closest crystal to the cursor
        closest_crystal_idx = -1
        min_dist = float('inf')
        
        for i, (cx, cy) in enumerate(self.crystals):
            dist = abs(cx - target_x) + abs(cy - target_y)
            if dist < min_dist:
                min_dist = dist
                closest_crystal_idx = i

        if closest_crystal_idx != -1:
            # Move the crystal
            self.crystals[closest_crystal_idx] = (target_x, target_y)
            # sfx: crystal move sound
            
            # Apply gravity
            self._apply_gravity()
            return True
        return False

    def _apply_gravity(self, spawn_particles=True):
        moved_in_pass = True
        while moved_in_pass:
            moved_in_pass = False
            # Sort by y-coordinate descending to process lower crystals first
            indices = sorted(range(len(self.crystals)), key=lambda k: self.crystals[k][1], reverse=True)
            
            crystal_pos_set = set(self.crystals)
            
            for i in indices:
                original_pos = self.crystals[i]
                cx, cy = original_pos
                
                # Check for support below
                is_supported = (cy + 1 >= self.GRID_HEIGHT) or \
                               (self.walls[cx][cy + 1] == 1) or \
                               ((cx, cy + 1) in crystal_pos_set and (cx, cy + 1) != original_pos)
                
                if not is_supported:
                    new_pos = (cx, cy + 1)
                    crystal_pos_set.remove(original_pos)
                    crystal_pos_set.add(new_pos)
                    self.crystals[i] = new_pos
                    moved_in_pass = True
                    # sfx: crystal landing sound
                    if spawn_particles:
                        self._spawn_particles(cx, cy + 1)

    def _calculate_alignment(self):
        if not self.crystals:
            return 0, 0

        total_pairs = 0
        max_chain = 0 if not self.crystals else 1

        grid = np.zeros((self.GRID_WIDTH, self.GRID_HEIGHT), dtype=int)
        for x, y in self.crystals:
            grid[x, y] = 1

        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if grid[x, y] == 1:
                    # Horizontal check
                    h_chain = 1
                    while x + h_chain < self.GRID_WIDTH and grid[x + h_chain, y] == 1:
                        h_chain += 1
                    if h_chain > 1: total_pairs += h_chain - 1
                    max_chain = max(max_chain, h_chain)
                    
                    # Vertical check
                    v_chain = 1
                    while y + v_chain < self.GRID_HEIGHT and grid[x, y + v_chain] == 1:
                        v_chain += 1
                    if v_chain > 1: total_pairs += v_chain - 1
                    max_chain = max(max_chain, v_chain)

        return max_chain, total_pairs // 2

    def _grid_to_iso(self, x, y):
        iso_x = self.WORLD_ORIGIN[0] + (x - y) * self.TILE_WIDTH_HALF
        iso_y = self.WORLD_ORIGIN[1] + (x + y) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _render_iso_tile(self, x, y, color):
        px, py = self._grid_to_iso(x, y)
        points = [
            (px, py - self.TILE_HEIGHT_HALF), (px + self.TILE_WIDTH_HALF, py),
            (px, py + self.TILE_HEIGHT_HALF), (px - self.TILE_WIDTH_HALF, py)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _render_iso_cube(self, x, y, top_color):
        self._render_iso_tile(x, y, top_color)

    def _render_game(self):
        render_queue = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.walls[x][y] == 1:
                    render_queue.append(('wall', x, y))
        for i, (cx, cy) in enumerate(self.crystals):
            render_queue.append(('crystal', cx, cy, i))

        render_queue.sort(key=lambda item: (item[2], item[1]))

        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if self.walls[x][y] == 0:
                    px, py = self._grid_to_iso(x, y)
                    points = [
                        (px, py - self.TILE_HEIGHT_HALF), (px + self.TILE_WIDTH_HALF, py),
                        (px, py + self.TILE_HEIGHT_HALF), (px - self.TILE_WIDTH_HALF, py)
                    ]
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        cursor_x, cursor_y = self.cursor_pos
        if self.walls[cursor_x][cursor_y] == 0:
            pulse = (math.sin(pygame.time.get_ticks() * 0.005) + 1) / 2
            cursor_color = tuple(np.clip([c + pulse * 30 for c in self.COLOR_CURSOR[:3]], 0, 255))
            self._render_iso_tile(cursor_x, cursor_y, cursor_color)

        for item in render_queue:
            if item[0] == 'wall':
                self._render_iso_cube(item[1], item[2], self.COLOR_WALL_TOP)
            elif item[0] == 'crystal':
                self._render_crystal(item[1], item[2], item[3])
        
        self._render_particles()

    def _render_crystal(self, x, y, index):
        px, py = self._grid_to_iso(x, y)
        color = self.CRYSTAL_COLORS[index % len(self.CRYSTAL_COLORS)]
        
        glow_radius = int(self.TILE_WIDTH_HALF * 0.8 + 3 * math.sin(pygame.time.get_ticks() * 0.001 + index))
        for i in range(glow_radius, 0, -2):
            alpha = int(50 * (1 - i / glow_radius))
            if alpha > 0:
                pygame.gfxdraw.filled_circle(self.screen, px, py, i, (*color, alpha))

        points = [
            (px, py - int(self.TILE_HEIGHT_HALF * 0.8)),
            (px + int(self.TILE_WIDTH_HALF * 0.5), py),
            (px, py + int(self.TILE_HEIGHT_HALF * 0.8)),
            (px - int(self.TILE_WIDTH_HALF * 0.5), py)
        ]
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

        pygame.gfxdraw.filled_circle(self.screen, px + int(self.TILE_WIDTH_HALF * 0.1), py - int(self.TILE_HEIGHT_HALF * 0.2), 2, (255, 255, 255))
        
    def _render_ui(self):
        moves_text = self.font_small.render(f"Moves Left: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (10, 10))
        
        align_text = self.font_small.render(f"Max Align: {self.max_align} / {self.WIN_ALIGNMENT}", True, self.COLOR_UI_TEXT)
        self.screen.blit(align_text, (10, 30))

        if self.game_over:
            overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            end_text_str = "ALIGNMENT COMPLETE" if self.max_align >= self.WIN_ALIGNMENT else "OUT OF MOVES"
            end_color = (150, 255, 150) if self.max_align >= self.WIN_ALIGNMENT else (255, 150, 150)
                
            end_text = self.font_large.render(end_text_str, True, end_color)
            text_rect = end_text.get_rect(center=self.screen.get_rect().center)
            self.screen.blit(end_text, text_rect)

    def _spawn_particles(self, grid_x, grid_y):
        px, py = self._grid_to_iso(grid_x, grid_y)
        py += self.TILE_HEIGHT_HALF
        for _ in range(self.np_random.integers(5, 10)):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.5, 1.5)
            self.particles.append({
                'pos': [px, py], 
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 1.0], 
                'life': self.np_random.integers(20, 40), 
                'max_life': 40, 
                'color': (self.np_random.integers(150, 255), self.np_random.integers(150, 255), self.np_random.integers(200, 255))
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.05
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _render_particles(self):
        for p in self.particles:
            life_ratio = max(0, p['life'] / p['max_life'])
            radius = int(3 * life_ratio)
            if radius > 0:
                color = (*p['color'], int(255 * life_ratio))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, color)

    def validate_implementation(self):
        print("✓ Running implementation validation...")
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    pygame.display.set_caption("Crystal Cavern")
    screen = pygame.display.set_mode((640, 400))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        if keys[pygame.K_SPACE]: action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: action[2] = 1

        obs, reward, terminated, truncated, info = env.step(action)
        
        if reward != 0:
            print(f"Reward: {reward:.2f}, Score: {info['score']:.2f}, Moves: {info['moves_left']}, Align: {info['max_alignment']}")

        if terminated or truncated:
            print("Game Over!")
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            pygame.time.wait(2000)
            obs, info = env.reset()

        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30)

    pygame.quit()