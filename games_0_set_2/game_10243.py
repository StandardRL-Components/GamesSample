import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T14:09:49.733311
# Source Brief: brief_00243.md
# Brief Index: 243
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Recreate target patterns on a hex grid by placing sand. Avoid detection by moving patrols while managing your limited sand resources."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place sand and hold shift to clone existing sand structures."
    )
    auto_advance = True

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    MAX_STEPS = 1000
    NUM_LEVELS = 10

    # --- Colors (Dreamlike Theme) ---
    COLOR_BG = (15, 10, 40)
    COLOR_GRID = (40, 30, 80)
    COLOR_TARGET = (70, 60, 110)
    COLOR_SAND = (0, 255, 255)
    COLOR_SAND_GLOW = (100, 255, 255)
    COLOR_CURSOR = (255, 255, 100)
    COLOR_CURSOR_GLOW = (255, 255, 180)
    COLOR_PATROL = (255, 20, 80)
    COLOR_PATROL_GLOW = (255, 80, 130)
    COLOR_PATROL_SCAN = (255, 20, 80, 50)
    COLOR_TEXT = (220, 220, 240)
    COLOR_SUCCESS = (50, 255, 150)
    COLOR_FAIL = (255, 50, 50)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_large = pygame.font.SysFont("Consolas", 48, bold=True)

        # --- Hex Grid Configuration ---
        self.hex_size = 18
        self.hex_grid_center = (self.SCREEN_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20)
        self.axial_directions = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        self.move_map = { 1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0) } # up, down, left, right
        self.grid_bounds = 10  # Max q/r coordinate

        # --- Game State (Persistent) ---
        self.unlocked_powers = {"clone"} # Start with cloning unlocked
        self.completed_levels = 0

        # --- Game State (Per-Episode) ---
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.current_level = 0
        self.grid = {}  # {(q, r): 'sand'}
        self.target_pattern = set()
        self.cursor_pos = (0, 0)
        self.sand_resources = 0
        self.patrols = []
        self.patrol_turn_period = 0
        self.patrol_turn_timer = 0
        self.particles = []
        self.last_move_dir = (1, 0)
        self.game_over_message = ""
        self.game_over_timer = 0

        # --- Visual Interpolation ---
        self.cursor_pixel_pos = np.array(self._hex_to_pixel(self.cursor_pos))
        
        self._define_puzzles_and_patrols()
        # self.reset() # reset is called by the wrapper/runner
        # self.validate_implementation() # No need to call this in production

    def _define_puzzles_and_patrols(self):
        self.puzzles = []
        self.patrol_paths = []
        # Level 0: Simple line
        self.puzzles.append({(0,0), (1,0), (2,0)})
        self.patrol_paths.append([[(5, -5), (5, 0), (0, 5), (-5, 5), (-5, 0), (0, -5)]])
        # Level 1: Triangle
        self.puzzles.append({(0,0), (1,0), (0,1)})
        self.patrol_paths.append([[(6, -3), (3, 3), (-3, 6), (-6, 3), (-3, -3), (3, -6)]])
        # Level 2: Larger Triangle
        self.puzzles.append({(0,0), (1,0), (2,0), (0,1), (1,1), (0,2)})
        self.patrol_paths.append([[(7,0), (0,7), (-7,7), (-7,0), (0,-7), (7,-7)], [(-4, -4), (4, -4), (4, 4), (-4, 4)]])
        # Add more levels...
        for i in range(3, self.NUM_LEVELS):
            puzzle = set()
            for _ in range(i + 3):
                puzzle.add((random.randint(-i//2, i//2), random.randint(-i//2, i//2)))
            self.puzzles.append(puzzle)
            
            path1 = []
            for j in range(6):
                angle = j * math.pi / 3
                path1.append((round(8 * math.cos(angle)), round(8 * math.sin(angle))))
            self.patrol_paths.append([path1])


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_over_timer = 0
        self.game_over_message = ""

        self.current_level = self.completed_levels % self.NUM_LEVELS
        
        self.target_pattern = self.puzzles[self.current_level]
        self.grid = {}
        self.sand_resources = len(self.target_pattern) + 3 # Extra sand
        self.cursor_pos = (0, 0)
        self.cursor_pixel_pos = np.array(self._hex_to_pixel(self.cursor_pos))
        self.particles = []

        # Difficulty scaling
        self.patrol_turn_period = max(2, 5 - self.completed_levels // 2)
        self.patrol_turn_timer = self.patrol_turn_period

        # Initialize patrols for the level
        self.patrols = []
        paths = self.patrol_paths[self.current_level]
        for path in paths:
            start_pos = path[0]
            self.patrols.append({
                'path': path,
                'index': 0,
                'pos': start_pos,
                'pixel_pos': np.array(self._hex_to_pixel(start_pos))
            })

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            self.game_over_timer -= 1
            if self.game_over_timer <= 0:
                # After a short delay, the episode truly ends.
                return self._get_observation(), 0, True, False, self._get_info()
            else:
                # Keep showing the game over screen.
                return self._get_observation(), 0, False, False, self._get_info()

        reward = 0
        terminated = False
        self.steps += 1

        movement, space_action, shift_action = action[0], action[1], action[2]
        
        # --- Handle Player Actions ---
        # 1. Movement
        if movement in self.move_map:
            move_dir = self.move_map[movement]
            self.last_move_dir = move_dir
            new_pos = (self.cursor_pos[0] + move_dir[0], self.cursor_pos[1] + move_dir[1])
            # Hexagonal distance check for bounds
            if max(abs(new_pos[0]), abs(new_pos[1]), abs(new_pos[0] + new_pos[1])) <= self.grid_bounds:
                self.cursor_pos = new_pos

        # 2. Actions (Clone has priority)
        if shift_action == 1 and 'clone' in self.unlocked_powers:
            reward += self._execute_clone()
        elif space_action == 1:
            reward += self._execute_place_sand()

        # --- Update Game Logic ---
        self.patrol_turn_timer -= 1
        if self.patrol_turn_timer <= 0:
            self.patrol_turn_timer = self.patrol_turn_period
            for patrol in self.patrols:
                # SFX: Patrol move swoosh
                patrol['index'] = (patrol['index'] + 1) % len(patrol['path'])
                patrol['pos'] = patrol['path'][patrol['index']]

        # --- Check Termination Conditions ---
        # 1. Detection by Sandman
        if self._check_detection():
            # SFX: Alarm, detection sound
            reward = -100
            self.game_over = True
            self.game_over_message = "DETECTED"
            self.game_over_timer = 60 # Frames to show message
            self.score += reward
            # Level progress resets on failure
            self.completed_levels = 0
            return self._get_observation(), reward, False, False, self._get_info()

        # 2. Puzzle Completion
        if self._check_puzzle_complete():
            # SFX: Success chime, level complete
            reward = 100
            self.game_over = True
            self.game_over_message = "COMPLETE"
            self.game_over_timer = 60 # Frames to show message
            self.score += reward
            self.completed_levels += 1
            # Potentially unlock powers here in a more complex game
            return self._get_observation(), reward, False, False, self._get_info()

        # 3. Max Steps
        if self.steps >= self.MAX_STEPS:
            reward = -100 # Penalty for timeout
            self.game_over = True
            self.game_over_message = "TIME OUT"
            self.game_over_timer = 60
            self.score += reward
            self.completed_levels = 0
            return self._get_observation(), reward, False, False, self._get_info()

        self.score += reward
        
        return self._get_observation(), reward, terminated, False, self._get_info()

    def _execute_clone(self):
        if self.cursor_pos not in self.grid:
            return 0 # Cannot clone empty space
        
        # 1. Find the structure to be cloned using BFS
        structure = set()
        q = [self.cursor_pos]
        visited = {self.cursor_pos}
        
        while q:
            pos = q.pop(0)
            structure.add(pos)
            for neighbor in self._get_hex_neighbors(pos):
                if neighbor in self.grid and neighbor not in visited:
                    visited.add(neighbor)
                    q.append(neighbor)
        
        # 2. Determine target location and check validity
        clone_cost = len(structure)
        if self.sand_resources < clone_cost:
            # SFX: Action failed sound
            return 0 # Not enough sand
        
        clone_offset = self.last_move_dir
        
        target_positions = {(pos[0] + clone_offset[0], pos[1] + clone_offset[1]) for pos in structure}

        # Check if target area is empty and within bounds
        for pos in target_positions:
            if pos in self.grid or max(abs(pos[0]), abs(pos[1]), abs(pos[0] + pos[1])) > self.grid_bounds:
                return 0 # Cannot clone onto existing sand or out of bounds

        # 3. Execute clone
        # SFX: Clone success sound
        self.sand_resources -= clone_cost
        for pos in target_positions:
            self.grid[pos] = 'sand'
            self._create_particles(self._hex_to_pixel(pos), self.COLOR_SAND, 15, 2)
        
        return 5.0

    def _execute_place_sand(self):
        if self.cursor_pos not in self.grid and self.sand_resources > 0:
            # SFX: Place sand puff
            self.grid[self.cursor_pos] = 'sand'
            self.sand_resources -= 1
            self._create_particles(self._hex_to_pixel(self.cursor_pos), self.COLOR_SAND, 10, 1.5)

            if self.cursor_pos in self.target_pattern:
                return 1.0 # Correct placement
            else:
                return -0.1 # Incorrect placement
        return 0 # No action taken

    def _check_detection(self):
        for patrol in self.patrols:
            patrol_neighbors = self._get_hex_neighbors(patrol['pos'])
            for neighbor in patrol_neighbors:
                if neighbor in self.grid:
                    return True
        return False

    def _check_puzzle_complete(self):
        placed_sand = set(self.grid.keys())
        return placed_sand == self.target_pattern

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._update_and_draw_visuals()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _update_and_draw_visuals(self):
        # --- Update positions for smooth interpolation ---
        target_cursor_pixel = np.array(self._hex_to_pixel(self.cursor_pos))
        self.cursor_pixel_pos += (target_cursor_pixel - self.cursor_pixel_pos) * 0.5

        for patrol in self.patrols:
            target_patrol_pixel = np.array(self._hex_to_pixel(patrol['pos']))
            patrol['pixel_pos'] += (target_patrol_pixel - patrol['pixel_pos']) * 0.25

        # --- Draw elements ---
        self._draw_hex_grid()
        self._draw_target_pattern()
        self._draw_sand()
        self._draw_patrols()
        self._draw_cursor()
        self._update_and_draw_particles()

        if self.game_over:
            self._draw_game_over_screen()

    def _draw_hex_grid(self):
        for q in range(-self.grid_bounds, self.grid_bounds + 1):
            for r in range(-self.grid_bounds, self.grid_bounds + 1):
                if max(abs(q), abs(r), abs(q + r)) <= self.grid_bounds:
                    center = self._hex_to_pixel((q, r))
                    self._draw_hexagon(self.screen, self.COLOR_GRID, center, self.hex_size, 1)

    def _draw_target_pattern(self):
        for pos in self.target_pattern:
            if pos not in self.grid:
                center = self._hex_to_pixel(pos)
                self._draw_hexagon(self.screen, self.COLOR_TARGET, center, self.hex_size, 2)
    
    def _draw_sand(self):
        for pos in self.grid:
            center = self._hex_to_pixel(pos)
            self._draw_glowing_hexagon(self.screen, self.COLOR_SAND, self.COLOR_SAND_GLOW, center, self.hex_size)

    def _draw_patrols(self):
        for patrol in self.patrols:
            center = patrol['pixel_pos']
            # Draw scan area
            scan_radius = self.hex_size * 2.5
            scan_surface = pygame.Surface((scan_radius * 2, scan_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(scan_surface, self.COLOR_PATROL_SCAN, (scan_radius, scan_radius), scan_radius)
            self.screen.blit(scan_surface, (center[0] - scan_radius, center[1] - scan_radius))
            # Draw patrol body
            self._draw_glowing_hexagon(self.screen, self.COLOR_PATROL, self.COLOR_PATROL_GLOW, center, self.hex_size * 0.8)

    def _draw_cursor(self):
        center = self.cursor_pixel_pos
        pulse = (math.sin(self.steps * 0.2) + 1) / 2
        size = self.hex_size * (1.1 + pulse * 0.2)
        self._draw_glowing_hexagon(self.screen, self.COLOR_CURSOR, self.COLOR_CURSOR_GLOW, center, size, width=3, glow_width=8)

    def _render_ui(self):
        sand_text = self.font_small.render(f"SAND: {self.sand_resources}", True, self.COLOR_TEXT)
        self.screen.blit(sand_text, (10, 10))

        level_text = self.font_small.render(f"LEVEL: {self.current_level + 1}/{self.NUM_LEVELS}", True, self.COLOR_TEXT)
        self.screen.blit(level_text, (self.SCREEN_WIDTH - level_text.get_width() - 10, 10))

        score_text = self.font_small.render(f"SCORE: {int(self.score)}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 30))

    def _draw_game_over_screen(self):
        overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
        color = self.COLOR_FAIL if "DETECTED" in self.game_over_message or "TIME" in self.game_over_message else self.COLOR_SUCCESS
        overlay.fill((color[0], color[1], color[2], 100))
        
        text_surface = self.font_large.render(self.game_over_message, True, self.COLOR_TEXT)
        text_rect = text_surface.get_rect(center=(self.SCREEN_WIDTH/2, self.SCREEN_HEIGHT/2))
        self.screen.blit(overlay, (0,0))
        self.screen.blit(text_surface, text_rect)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.current_level}

    # --- Helper & Drawing Functions ---
    def _hex_to_pixel(self, hex_coord):
        q, r = hex_coord
        x = self.hex_size * (3/2 * q)
        y = self.hex_size * (math.sqrt(3)/2 * q + math.sqrt(3) * r)
        return (x + self.hex_grid_center[0], y + self.hex_grid_center[1])

    def _get_hex_neighbors(self, hex_coord):
        q, r = hex_coord
        return [(q + dq, r + dr) for dq, dr in self.axial_directions]

    def _draw_hexagon(self, surface, color, center, size, width=0):
        points = []
        for i in range(6):
            angle = math.pi / 3 * i + math.pi / 6
            points.append((center[0] + size * math.cos(angle), 
                           center[1] + size * math.sin(angle)))
        if width == 0:
            pygame.gfxdraw.filled_polygon(surface, [(int(p[0]), int(p[1])) for p in points], color)
        else:
            pygame.gfxdraw.aapolygon(surface, [(int(p[0]), int(p[1])) for p in points], color)
            if width > 1: # Thicken line
                 for i in range(1, width):
                    points_inner = [(center[0] + (size-i) * math.cos(math.pi / 3 * j + math.pi / 6),
                                     center[1] + (size-i) * math.sin(math.pi / 3 * j + math.pi / 6)) for j in range(6)]
                    pygame.gfxdraw.aapolygon(surface, [(int(p[0]), int(p[1])) for p in points_inner], color)

    def _draw_glowing_hexagon(self, surface, color, glow_color, center, size, width=0, glow_width=10):
        # Draw multiple layers for glow
        for i in range(glow_width, 0, -2):
            s = size + i
            alpha = 1 - (i / glow_width)
            c = (*glow_color, int(alpha * 50))
            self._draw_hexagon(surface, c, center, s)
        # Draw main shape
        self._draw_hexagon(surface, color, center, size, width)

    def _create_particles(self, pos, color, count, speed):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            vel = [math.cos(angle) * speed * random.uniform(0.5, 1.5),
                   math.sin(angle) * speed * random.uniform(0.5, 1.5)]
            self.particles.append({
                'pos': list(pos),
                'vel': vel,
                'lifespan': random.randint(20, 40),
                'color': color
            })

    def _update_and_draw_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            
            # Fade out
            alpha = max(0, p['lifespan'] / 40)
            radius = int(alpha * 3)
            if radius > 0:
                pygame.draw.circle(self.screen, (*p['color'], int(alpha*255)), 
                                   [int(p['pos'][0]), int(p['pos'][1])], radius)

        self.particles = [p for p in self.particles if p['lifespan'] > 0]
    
    def close(self):
        pygame.quit()

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (400, 640, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (400, 640, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (400, 640, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

# Example of how to run the environment
if __name__ == '__main__':
    # The main execution block is for manual testing and visualization.
    # It requires a display, so we unset the dummy video driver.
    if "SDL_VIDEODRIVER" in os.environ:
        del os.environ["SDL_VIDEODRIVER"]

    env = GameEnv(render_mode="rgb_array")
    
    # --- Manual Play ---
    pygame.display.set_caption("Sandman's Dreamscapes - Manual Control")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    
    obs, info = env.reset()
    done = False
    
    # Game loop for manual play
    while not done:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"Episode finished. Score: {info['score']}")
            # The env handles the game over screen delay, so we wait for the real 'done' signal
            if terminated:
                # Wait for the game over screen to finish
                real_done = False
                while not real_done:
                    obs, reward, terminated_final, truncated_final, info = env.step([0,0,0])
                    # Update screen
                    surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    if terminated_final: # Real termination signal
                        real_done = True
                
                # Reset for next game
                obs, info = env.reset()
                # done = False # Uncomment to play multiple games

        # Update screen
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        env.clock.tick(30) # Limit to 30 FPS for smooth viewing

    env.close()