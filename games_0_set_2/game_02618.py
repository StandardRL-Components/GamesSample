
# Generated: 2025-08-27T20:55:02.033352
# Source Brief: brief_02618.md
# Brief Index: 2618

        
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
    """
    An isometric puzzle game where the player manipulates crystals to match target patterns.
    The game is turn-based, with a limited number of moves per puzzle.
    Visuals are a key focus, with glowing crystals and a clean, geometric aesthetic.
    """
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the selector. Press Space to pick up a crystal "
        "and move it to an empty adjacent tile in the direction of your last move. "
        "Hold Shift to restart the current puzzle."
    )

    game_description = (
        "A turn-based isometric puzzle game. Arrange glowing crystals to match the target "
        "pattern shown on the right. Solve all 15 puzzles with a limited number of moves "
        "to win. Planning your moves carefully is key to success!"
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 5
        self.TILE_WIDTH = 48
        self.TILE_HEIGHT = self.TILE_WIDTH // 2
        self.MAX_STEPS = 1500

        # --- Colors ---
        self.COLOR_BG = (15, 18, 32)
        self.COLOR_GRID = (30, 36, 64)
        self.COLOR_CURSOR = (255, 255, 100)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 20)
        self.CRYSTAL_COLORS = [
            ((0, 180, 255), (0, 120, 200)),  # Blue
            ((255, 80, 120), (200, 50, 80)),   # Red
            ((80, 255, 150), (50, 200, 100)),  # Green
            ((255, 180, 0), (200, 140, 0)),   # Orange
            ((200, 100, 255), (150, 70, 200)), # Purple
        ]
        self.TARGET_GHOST_ALPHA = 60

        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.Font(None, 32)
        self.font_small = pygame.font.Font(None, 24)
        self.font_title = pygame.font.Font(None, 48)

        # --- Game State Initialization ---
        self.puzzles = self._generate_puzzles()
        self.grid_offset_x = (self.WIDTH - self.GRID_SIZE * self.TILE_WIDTH) // 2 - 100
        self.grid_offset_y = (self.HEIGHT - self.GRID_SIZE * self.TILE_HEIGHT) // 2 + 50
        
        # These will be initialized in reset()
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.current_puzzle_index = 0
        self.crystals = []
        self.target_layout = {}
        self.moves_left = 0
        self.moves_total = 0
        self.cursor_pos = [0, 0]
        self.last_move_dir = [0, 0]
        self.particles = []

        self.reset()
        self.validate_implementation()

    def _generate_puzzles(self):
        # Puzzles are defined as dictionaries:
        # 'start': list of (x, y, color_idx) for initial crystal positions
        # 'target': list of (x, y, color_idx) for target positions
        # 'moves': number of allowed moves
        # NOTE: The i-th crystal in 'start' corresponds to the i-th position in 'target'
        return [
            {'start': [(2, 2, 0)], 'target': [(2, 1, 0)], 'moves': 1},
            {'start': [(1, 2, 0), (3, 2, 1)], 'target': [(2, 1, 0), (2, 3, 1)], 'moves': 4},
            {'start': [(0, 0, 0), (4, 4, 1)], 'target': [(4, 0, 0), (0, 4, 1)], 'moves': 4},
            {'start': [(1, 1, 0), (1, 3, 1), (3, 1, 2)], 'target': [(3, 3, 0), (1, 1, 1), (3, 1, 2)], 'moves': 3},
            {'start': [(0, 2, 0), (2, 0, 1), (4, 2, 2)], 'target': [(2, 4, 0), (0, 2, 1), (2, 0, 2)], 'moves': 6},
            {'start': [(1, 1, 0), (1, 2, 1), (1, 3, 2)], 'target': [(3, 1, 0), (3, 2, 1), (3, 3, 2)], 'moves': 6},
            {'start': [(0, 0, 0), (0, 4, 1), (4, 0, 2), (4, 4, 3)], 'target': [(2, 1, 0), (1, 2, 1), (3, 2, 2), (2, 3, 3)], 'moves': 8},
            {'start': [(2, 1, 0), (1, 2, 1), (3, 2, 2), (2, 3, 3)], 'target': [(1, 1, 0), (1, 3, 1), (3, 1, 2), (3, 3, 3)], 'moves': 8},
            {'start': [(0, 2, 0), (1, 2, 1), (3, 2, 2), (4, 2, 3)], 'target': [(2, 0, 0), (2, 1, 1), (2, 3, 2), (2, 4, 3)], 'moves': 12},
            {'start': [(1, 1, 0), (1, 3, 1), (3, 1, 2), (3, 3, 3)], 'target': [(1, 2, 0), (2, 1, 1), (2, 3, 2), (3, 2, 3)], 'moves': 4},
            {'start': [(0,0,0), (0,1,1), (1,0,2), (1,1,3)], 'target': [(3,3,0), (3,4,1), (4,3,2), (4,4,3)], 'moves': 12},
            {'start': [(2,0,0),(2,1,1),(2,3,2),(2,4,3)], 'target': [(0,2,0),(1,2,1),(3,2,2),(4,2,3)], 'moves': 12},
            {'start': [(0,0,0),(1,1,1),(2,2,2),(3,3,3),(4,4,4)], 'target': [(4,0,0),(3,1,1),(2,2,2),(1,3,3),(0,4,4)], 'moves': 8},
            {'start': [(0,2,0),(2,0,1),(2,4,2),(4,2,3)], 'target': [(1,1,0),(1,3,1),(3,1,2),(3,3,3)], 'moves': 8},
            {'start': [(0,1,0),(1,3,1),(3,1,2),(4,3,3)], 'target': [(1,0,0),(3,0,1),(1,4,2),(3,4,3)], 'moves': 12},
        ]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.current_puzzle_index = 0
        self._load_puzzle(self.current_puzzle_index)
        return self._get_observation(), self._get_info()

    def _load_puzzle(self, puzzle_index):
        if puzzle_index >= len(self.puzzles):
            self.game_over = True
            return

        puzzle_data = self.puzzles[puzzle_index]
        self.moves_total = puzzle_data['moves']
        self.moves_left = self.moves_total
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.last_move_dir = [0, 0]
        self.particles = []

        self.crystals = []
        self.target_layout = {}
        
        start_positions = puzzle_data['start']
        target_positions = puzzle_data['target']

        for i in range(len(start_positions)):
            sx, sy, c_idx = start_positions[i]
            tx, ty, _ = target_positions[i]
            self.crystals.append({
                'pos': [sx, sy],
                'target_pos': [tx, ty],
                'color_idx': c_idx,
            })
            self.target_layout[(tx, ty)] = c_idx

    def _get_dist_from_target(self):
        dist = 0
        for c in self.crystals:
            dist += abs(c['pos'][0] - c['target_pos'][0]) + abs(c['pos'][1] - c['target_pos'][1])
        return dist

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action
        reward = 0
        
        # --- Action: Restart Puzzle ---
        if shift_held:
            self._load_puzzle(self.current_puzzle_index)
            # Small penalty for restarting
            reward = -1
            return self._get_observation(), reward, self.game_over, False, self._get_info()

        # --- Action: Move Cursor ---
        if movement != 0:
            dx, dy = [(0,0), (0,-1), (0,1), (-1,0), (1,0)][movement]
            self.cursor_pos[0] = np.clip(self.cursor_pos[0] + dx, 0, self.GRID_SIZE - 1)
            self.cursor_pos[1] = np.clip(self.cursor_pos[1] + dy, 0, self.GRID_SIZE - 1)
            self.last_move_dir = [dx, dy]

        # --- Action: Move Crystal ---
        if space_held and self.last_move_dir != [0, 0] and self.moves_left > 0:
            crystal_to_move = None
            for c in self.crystals:
                if c['pos'] == self.cursor_pos:
                    crystal_to_move = c
                    break
            
            if crystal_to_move:
                target_x = crystal_to_move['pos'][0] + self.last_move_dir[0]
                target_y = crystal_to_move['pos'][1] + self.last_move_dir[1]

                # Check if target is valid (in bounds and empty)
                is_in_bounds = 0 <= target_x < self.GRID_SIZE and 0 <= target_y < self.GRID_SIZE
                is_empty = not any(c['pos'] == [target_x, target_y] for c in self.crystals)

                if is_in_bounds and is_empty:
                    # Sound effect placeholder: # sfx_crystal_slide()
                    dist_before = self._get_dist_from_target()
                    
                    old_pos_screen = self._world_to_iso(*crystal_to_move['pos'])
                    crystal_to_move['pos'] = [target_x, target_y]
                    self.moves_left -= 1
                    
                    dist_after = self._get_dist_from_target()
                    reward += (dist_before - dist_after) # Continuous reward
                    
                    # Add particle effect
                    new_pos_screen = self._world_to_iso(*crystal_to_move['pos'])
                    self._create_move_particles(old_pos_screen, new_pos_screen, crystal_to_move['color_idx'])


        # --- Check for Puzzle Completion ---
        if self._check_puzzle_completion():
            # Sound effect placeholder: # sfx_puzzle_complete()
            puzzle_reward = 10 + self.moves_left # Bonus for efficiency
            reward += puzzle_reward
            self.score += puzzle_reward
            self.current_puzzle_index += 1
            if self.current_puzzle_index >= len(self.puzzles):
                # Game won!
                self.game_over = True
                reward += 50 # Final completion bonus
            else:
                self._load_puzzle(self.current_puzzle_index)

        # --- Check for Failure (out of moves) ---
        if self.moves_left <= 0 and not self._check_puzzle_completion():
            # Sound effect placeholder: # sfx_puzzle_fail()
            self.game_over = True
        
        self.steps += 1
        if self.steps >= self.MAX_STEPS:
            self.game_over = True
            
        return self._get_observation(), reward, self.game_over, False, self._get_info()

    def _check_puzzle_completion(self):
        if not self.crystals:
            return False
        for c in self.crystals:
            if c['pos'] != c['target_pos']:
                return False
        return True

    def _world_to_iso(self, x, y):
        iso_x = self.grid_offset_x + (x - y) * (self.TILE_WIDTH / 2)
        iso_y = self.grid_offset_y + (x + y) * (self.TILE_HEIGHT / 2)
        return int(iso_x), int(iso_y)

    def _render_iso_cube(self, surface, x, y, color, highlight_color, height=20):
        iso_x, iso_y = self._world_to_iso(x, y)
        
        # Points for the cube
        p_top = (iso_x, iso_y - height)
        p_bottom = (iso_x, iso_y)
        p_left = (iso_x - self.TILE_WIDTH / 2, iso_y - self.TILE_HEIGHT / 2)
        p_right = (iso_x + self.TILE_WIDTH / 2, iso_y - self.TILE_HEIGHT / 2)
        
        p_top_left = (iso_x - self.TILE_WIDTH / 2, iso_y - height - self.TILE_HEIGHT / 2)
        p_top_right = (iso_x + self.TILE_WIDTH / 2, iso_y - height - self.TILE_HEIGHT / 2)

        # Draw faces
        # Top face
        pygame.draw.polygon(surface, highlight_color, [p_top, p_top_right, (iso_x, iso_y - height + self.TILE_HEIGHT / 2), p_top_left])
        # Left face
        pygame.draw.polygon(surface, color, [p_top_left, (iso_x, iso_y - height + self.TILE_HEIGHT / 2), p_bottom, p_left])
        # Right face
        pygame.draw.polygon(surface, color, [p_top_right, (iso_x, iso_y - height + self.TILE_HEIGHT / 2), p_bottom, p_right])
        
        # Glow effect
        glow_radius = int(self.TILE_WIDTH * 0.6)
        for i in range(glow_radius, 0, -2):
            alpha = 60 * (1 - i / glow_radius)
            pygame.gfxdraw.filled_circle(surface, int(iso_x), int(iso_y - height / 2), i, (*highlight_color, alpha))

    def _render_iso_tile(self, surface, x, y, color):
        iso_x, iso_y = self._world_to_iso(x, y)
        points = [
            (iso_x, iso_y),
            (iso_x + self.TILE_WIDTH / 2, iso_y + self.TILE_HEIGHT / 2),
            (iso_x, iso_y + self.TILE_HEIGHT),
            (iso_x - self.TILE_WIDTH / 2, iso_y + self.TILE_HEIGHT / 2)
        ]
        pygame.draw.polygon(surface, color, points)

    def _create_move_particles(self, start_pos, end_pos, color_idx):
        color, _ = self.CRYSTAL_COLORS[color_idx]
        for _ in range(20):
            self.particles.append({
                'pos': list(start_pos),
                'vel': [(end_pos[0] - start_pos[0]) / 10 + random.uniform(-1, 1), 
                        (end_pos[1] - start_pos[1]) / 10 + random.uniform(-1, 1)],
                'life': random.randint(10, 20),
                'radius': random.uniform(2, 5),
                'color': color
            })

    def _update_and_render_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
            p['radius'] *= 0.95
            if p['life'] <= 0 or p['radius'] < 1:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 20))
                pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), int(p['radius']), (*p['color'], alpha))

    def _render_text(self, text, font, x, y, color=None, shadow=True):
        if color is None: color = self.COLOR_TEXT
        if shadow:
            text_surf_shadow = font.render(text, True, self.COLOR_TEXT_SHADOW)
            self.screen.blit(text_surf_shadow, (x + 2, y + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, (x, y))

    def _render_game(self):
        # Draw grid floor
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                self._render_iso_tile(self.screen, r, c, self.COLOR_GRID)

        # Draw target "ghosts"
        target_surface = self.screen.copy()
        target_surface.set_colorkey(self.COLOR_BG)
        target_surface.fill(self.COLOR_BG)
        for (tx, ty), c_idx in self.target_layout.items():
            color, highlight = self.CRYSTAL_COLORS[c_idx]
            self._render_iso_cube(target_surface, tx, ty, color, highlight)
        target_surface.set_alpha(self.TARGET_GHOST_ALPHA)
        self.screen.blit(target_surface, (0, 0))

        # Draw cursor
        self._render_iso_tile(self.screen, self.cursor_pos[0], self.cursor_pos[1], self.COLOR_CURSOR)

        # Draw crystals
        for crystal in sorted(self.crystals, key=lambda c: c['pos'][0] + c['pos'][1]):
            color, highlight = self.CRYSTAL_COLORS[crystal['color_idx']]
            self._render_iso_cube(self.screen, crystal['pos'][0], crystal['pos'][1], color, highlight)
        
        self._update_and_render_particles()

    def _render_ui(self):
        # --- Main Info Panel (Top Left) ---
        self._render_text(f"Puzzle: {self.current_puzzle_index + 1} / {len(self.puzzles)}", self.font_main, 20, 20)
        self._render_text(f"Score: {self.score}", self.font_main, 20, 50)
        self._render_text(f"Moves Left: {self.moves_left}", self.font_main, 20, 80)
        
        # --- Target Pattern Display (Right) ---
        self._render_text("Target", self.font_main, 480, 100)
        target_offset_x = 480
        target_offset_y = 220
        
        # Draw a mini-grid for the target
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                points = [
                    (target_offset_x + (r-c)*12, target_offset_y + (r+c)*6),
                    (target_offset_x + (r-c)*12 + 12, target_offset_y + (r+c)*6 + 6),
                    (target_offset_x + (r-c)*12, target_offset_y + (r+c)*6 + 12),
                    (target_offset_x + (r-c)*12 - 12, target_offset_y + (r+c)*6 + 6),
                ]
                pygame.draw.polygon(self.screen, self.COLOR_GRID, points)

        # Draw mini target crystals
        for (tx, ty), c_idx in self.target_layout.items():
            color, highlight = self.CRYSTAL_COLORS[c_idx]
            iso_x = target_offset_x + (tx - ty) * 12
            iso_y = target_offset_y + (tx + ty) * 6
            pygame.draw.circle(self.screen, color, (iso_x, iso_y), 8)
            pygame.draw.circle(self.screen, highlight, (iso_x, iso_y), 5)

        # --- Game Over / Win Message ---
        if self.game_over:
            s = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            s.fill((0,0,0,180))
            self.screen.blit(s, (0,0))
            if self._check_puzzle_completion():
                self._render_text("ALL PUZZLES SOLVED!", self.font_title, self.WIDTH/2 - 220, self.HEIGHT/2 - 50)
                self._render_text(f"Final Score: {self.score}", self.font_main, self.WIDTH/2 - 90, self.HEIGHT/2 + 10)
            else:
                self._render_text("OUT OF MOVES", self.font_title, self.WIDTH/2 - 150, self.HEIGHT/2 - 50)
                self._render_text("Press SHIFT to restart puzzle", self.font_main, self.WIDTH/2 - 160, self.HEIGHT/2 + 10)


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
            "puzzle": self.current_puzzle_index + 1,
            "moves_left": self.moves_left,
            "is_complete": self._check_puzzle_completion()
        }

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    running = True
    pygame.display.set_caption("Crystal Caverns")
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    clock = pygame.time.Clock()

    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space_held = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

        action = [movement, space_held, shift_held]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # We need to send an action even for key releases to advance state
            if event.type in [pygame.KEYDOWN, pygame.KEYUP]:
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated:
                    print(f"Game Over! Final Score: {info['score']}")
                    # Optional: reset after a delay or key press
                    # pygame.time.wait(2000)
                    # obs, info = env.reset()

        # In a manual play loop, we need to continuously render
        # but only step on input. This is a bit different from the auto_advance=False logic
        # for an agent, but works for human play.
        # So we just get the latest observation without stepping again.
        frame = env._get_observation()
        
        # Pygame uses (width, height), but our obs is (height, width, 3). Transpose for display.
        surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate

    env.close()