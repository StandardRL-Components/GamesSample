
# Generated: 2025-08-28T06:31:20.631728
# Source Brief: brief_02947.md
# Brief Index: 2947

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
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

    user_guide = (
        "Controls: Arrows to move cursor. Space to select/deselect a crystal. "
        "Arrows to push a selected crystal. Shift to deselect."
    )

    game_description = (
        "A turn-based puzzle game. Push glowing crystals to match the target pattern "
        "on the right before you run out of moves. Plan your pushes carefully, as "
        "they can cause chain reactions."
    )

    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Spaces ---
        self.observation_space = Box(low=0, high=255, shape=(400, 640, 3), dtype=np.uint8)
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.WIDTH, self.HEIGHT = 640, 400
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()

        # --- Game Constants ---
        self.GRID_SIZE = 7
        self.MAX_MOVES = 30
        self.MAX_STEPS = 1000
        self.NUM_CRYSTAL_TYPES = 4
        self.INITIAL_CRYSTALS = 5
        self.SCRAMBLE_MOVES = 15

        # --- Visuals ---
        self.font_large = pygame.font.SysFont("Consolas", 24, bold=True)
        self.font_medium = pygame.font.SysFont("Consolas", 18, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)
        
        self.COLOR_BG = (15, 20, 35)
        self.COLOR_GRID = (40, 50, 70)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 20)
        self.COLOR_CURSOR = (255, 255, 100, 100)
        self.COLOR_SELECTED = (255, 255, 255, 150)
        
        self.CRYSTAL_COLORS = [
            (0, 255, 255),  # Cyan
            (255, 0, 255),  # Magenta
            (255, 255, 0),  # Yellow
            (0, 255, 128),  # Spring Green
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (255, 64, 64),  # Red
            (64, 64, 255),  # Blue
            (255, 255, 255),# White
            (128, 255, 0),  # Lime
        ]

        # Isometric projection parameters
        self.tile_width_half = 26
        self.tile_height_half = 13
        self.origin_x = self.WIDTH // 2 - 120
        self.origin_y = 90
        
        # --- State Variables ---
        self.grid = {}
        self.target_grid = {}
        self.target_map = {}
        self.cursor_pos = (0, 0)
        self.selected_pos = None
        self.remaining_moves = 0
        self.particles = []
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.last_space_held = False
        self.last_reward_info = ""

        # Initialize state
        self.reset()
        
        # Run validation check
        # self.validate_implementation() # Comment out for submission

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.remaining_moves = self.MAX_MOVES
        self.cursor_pos = (self.GRID_SIZE // 2, self.GRID_SIZE // 2)
        self.selected_pos = None
        self.particles.clear()
        self.last_space_held = False
        self.last_reward_info = ""

        self.grid, self.target_grid, self.target_map = self._generate_puzzle()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement, space_held, shift_held = action[0], action[1] == 1, action[2] == 1
        reward = 0
        action_taken = False

        # Debounce spacebar to register presses, not holds
        space_pressed = space_held and not self.last_space_held
        self.last_space_held = space_held
        
        # --- Handle Actions ---
        # 1. Deselection with Shift
        if shift_held and self.selected_pos:
            self.selected_pos = None
            # sfx: deselect_sound

        # 2. Selection/Deselection with Space
        elif space_pressed:
            if self.selected_pos is not None:
                self.selected_pos = None
                # sfx: deselect_sound
            elif self.cursor_pos in self.grid:
                self.selected_pos = self.cursor_pos
                # sfx: select_sound

        # 3. Push Action (if a crystal is selected and movement is attempted)
        elif self.selected_pos and movement != 0:
            action_taken = True
            reward, move_info = self._execute_push(movement)
            self.last_reward_info = move_info
            self.selected_pos = None # Deselect after push
            # sfx: push_sound
        
        # 4. Cursor Movement (if no crystal is selected)
        elif not self.selected_pos and movement != 0:
            dx, dy = self._get_direction_vector(movement)
            self.cursor_pos = (
                max(0, min(self.GRID_SIZE - 1, self.cursor_pos[0] + dx)),
                max(0, min(self.GRID_SIZE - 1, self.cursor_pos[1] + dy))
            )
            # sfx: cursor_move_sound

        if action_taken:
            self.remaining_moves -= 1
            self.score += reward

        # --- Update Game State ---
        self._update_particles()
        self.steps += 1
        
        # --- Check Termination ---
        is_win = self._check_win()
        is_loss = self.remaining_moves <= 0
        terminated = is_win or is_loss or self.steps >= self.MAX_STEPS

        if terminated and not self.game_over:
            self.game_over = True
            if is_win:
                reward += 100
                self.score += 100
                self.last_reward_info = "PUZZLE SOLVED!"
                # sfx: win_jingle
            else:
                reward -= 100
                self.score -= 100
                self.last_reward_info = "OUT OF MOVES"
                # sfx: lose_sound

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _execute_push(self, movement):
        direction = self._get_direction_vector(movement)
        start_pos = self.selected_pos
        
        # Calculate pre-move distances for reward
        dist_before = self._calculate_target_distance(self.grid)

        # Build the chain of crystals to be pushed
        chain = []
        current_pos = start_pos
        while current_pos in self.grid:
            chain.append(current_pos)
            current_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
        
        # Check if the push is valid (doesn't go out of bounds)
        end_pos = current_pos
        if not (0 <= end_pos[0] < self.GRID_SIZE and 0 <= end_pos[1] < self.GRID_SIZE):
            # sfx: invalid_move_sound
            return 0, "INVALID MOVE"

        # Execute the push by moving crystals in reverse order of the chain
        new_grid = self.grid.copy()
        moved_crystals = []
        for pos in reversed(chain):
            crystal_type = new_grid.pop(pos)
            new_pos = (pos[0] + direction[0], pos[1] + direction[1])
            new_grid[new_pos] = crystal_type
            moved_crystals.append((new_pos, crystal_type))
        self.grid = new_grid
        
        # sfx: crystal_clack_sound
        for pos, c_type in moved_crystals:
            self._create_particles(pos, self.CRYSTAL_COLORS[c_type])

        # --- Calculate Reward ---
        dist_after = self._calculate_target_distance(self.grid)
        reward = dist_before - dist_after # Positive if closer, negative if further

        placed_correctly = 0
        for pos, c_type in moved_crystals:
            if pos in self.target_map and self.target_map[pos] == c_type:
                placed_correctly += 1
                reward += 5

        move_info = f"Δ Dist: {reward:+.0f}"
        if placed_correctly > 0:
            move_info += f", Placed: {placed_correctly}"

        return reward, move_info

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "moves_left": self.remaining_moves}

    def _render_game(self):
        # Draw grid
        for i in range(self.GRID_SIZE + 1):
            # Horizontal lines
            p1 = self._cart_to_iso(i, 0)
            p2 = self._cart_to_iso(i, self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
            # Vertical lines
            p1 = self._cart_to_iso(0, i)
            p2 = self._cart_to_iso(self.GRID_SIZE, i)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

        # Draw particles
        for p in self.particles:
            iso_pos = self._cart_to_iso(p['pos'][0], p['pos'][1])
            alpha = max(0, 255 * (p['life'] / p['max_life']))
            color = (*p['color'], alpha)
            temp_surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (p['size'], p['size']), p['size'])
            self.screen.blit(temp_surf, (iso_pos[0] - p['size'], iso_pos[1] - p['size']))

        # Draw crystals
        sorted_grid_keys = sorted(self.grid.keys(), key=lambda p: p[0] + p[1])
        for x, y in sorted_grid_keys:
            color = self.CRYSTAL_COLORS[self.grid[(x, y)]]
            is_selected = self.selected_pos == (x, y)
            self._draw_iso_cube((x, y), color, is_selected)

        # Draw cursor
        if not self.selected_pos:
            cursor_surf = pygame.Surface((self.tile_width_half * 2, self.tile_height_half * 2), pygame.SRCALPHA)
            points = [
                (self.tile_width_half, 0),
                (self.tile_width_half * 2, self.tile_height_half),
                (self.tile_width_half, self.tile_height_half * 2),
                (0, self.tile_height_half)
            ]
            pygame.gfxdraw.filled_polygon(cursor_surf, points, self.COLOR_CURSOR)
            iso_pos = self._cart_to_iso(self.cursor_pos[0], self.cursor_pos[1])
            self.screen.blit(cursor_surf, (iso_pos[0] - self.tile_width_half, iso_pos[1] - self.tile_height_half))
    
    def _render_ui(self):
        # --- Target Pattern Display ---
        self._draw_text("TARGET", (self.WIDTH - 120, 30), self.font_medium)
        target_origin_x, target_origin_y = self.WIDTH - 120, 100
        target_tile_w, target_tile_h = 10, 5
        
        for x in range(self.GRID_SIZE):
            for y in range(self.GRID_SIZE):
                iso_x = target_origin_x + (x - y) * target_tile_w
                iso_y = target_origin_y + (x + y) * target_tile_h
                points = [
                    (iso_x, iso_y - target_tile_h),
                    (iso_x + target_tile_w, iso_y),
                    (iso_x, iso_y + target_tile_h),
                    (iso_x - target_tile_w, iso_y)
                ]
                if (x, y) in self.target_grid:
                    color = self.CRYSTAL_COLORS[self.target_grid[(x, y)]]
                    pygame.gfxdraw.filled_polygon(self.screen, points, color)
                    pygame.gfxdraw.aapolygon(self.screen, points, color)
                else:
                    pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID)

        # --- Info Text ---
        self._draw_text(f"MOVES: {self.remaining_moves}", (20, 20), self.font_large)
        self._draw_text(f"SCORE: {int(self.score)}", (20, 50), self.font_large)
        if self.last_reward_info:
            self._draw_text(self.last_reward_info, (self.WIDTH // 2, 20), self.font_medium, center=True)
        
        if self.game_over:
            msg = "PUZZLE SOLVED!" if self._check_win() else "OUT OF MOVES"
            self._draw_text(msg, (self.WIDTH // 2, self.HEIGHT // 2 - 20), self.font_large, center=True)
            self._draw_text("Press RESET to play again", (self.WIDTH // 2, self.HEIGHT // 2 + 20), self.font_medium, center=True)

    def _draw_text(self, text, pos, font, color=None, center=False):
        color = color or self.COLOR_TEXT
        shadow_surface = font.render(text, True, self.COLOR_TEXT_SHADOW)
        text_surface = font.render(text, True, color)
        
        text_rect = text_surface.get_rect()
        if center:
            text_rect.center = pos
        else:
            text_rect.topleft = pos

        self.screen.blit(shadow_surface, (text_rect.x + 2, text_rect.y + 2))
        self.screen.blit(text_surface, text_rect)

    def _cart_to_iso(self, x, y):
        iso_x = self.origin_x + (x - y) * self.tile_width_half
        iso_y = self.origin_y + (x + y) * self.tile_height_half
        return int(iso_x), int(iso_y)

    def _draw_iso_cube(self, pos, color, is_selected):
        x, y = pos
        iso_x, iso_y = self._cart_to_iso(x, y)
        
        height = self.tile_height_half * 2
        
        top_face = [
            (iso_x, iso_y - height),
            (iso_x + self.tile_width_half, iso_y - self.tile_height_half),
            (iso_x, iso_y),
            (iso_x - self.tile_width_half, iso_y - self.tile_height_half)
        ]
        
        left_face = [
            (iso_x - self.tile_width_half, iso_y - self.tile_height_half),
            (iso_x, iso_y),
            (iso_x, iso_y + height),
            (iso_x - self.tile_width_half, iso_y + self.tile_height_half)
        ]

        right_face = [
            (iso_x + self.tile_width_half, iso_y - self.tile_height_half),
            (iso_x, iso_y),
            (iso_x, iso_y + height),
            (iso_x + self.tile_width_half, iso_y + self.tile_height_half)
        ]

        c_light = tuple(min(255, c + 40) for c in color)
        c_dark = tuple(max(0, c - 40) for c in color)

        if is_selected:
            # Draw a pulsing glow
            pulse = abs(math.sin(pygame.time.get_ticks() * 0.005))
            glow_size = int(self.tile_width_half * (1.2 + pulse * 0.2))
            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*self.COLOR_SELECTED[:3], 50 + int(pulse * 50)), (glow_size, glow_size), glow_size)
            self.screen.blit(glow_surf, (iso_x - glow_size, iso_y - glow_size))

        pygame.gfxdraw.filled_polygon(self.screen, top_face, c_light)
        pygame.gfxdraw.filled_polygon(self.screen, left_face, c_dark)
        pygame.gfxdraw.filled_polygon(self.screen, right_face, color)
        
        pygame.gfxdraw.aapolygon(self.screen, top_face, c_light)
        pygame.gfxdraw.aapolygon(self.screen, left_face, c_dark)
        pygame.gfxdraw.aapolygon(self.screen, right_face, color)

    def _generate_puzzle(self):
        # 1. Create a solvable target state
        target_grid = {}
        available_pos = [(x, y) for x in range(self.GRID_SIZE) for y in range(self.GRID_SIZE)]
        self.np_random.shuffle(available_pos)
        
        crystal_types = list(range(self.NUM_CRYSTAL_TYPES))
        for i in range(self.INITIAL_CRYSTALS):
            pos = available_pos.pop()
            c_type = self.np_random.choice(crystal_types)
            target_grid[pos] = c_type
        
        # 2. Scramble the grid by performing random pushes
        grid = target_grid.copy()
        for _ in range(self.SCRAMBLE_MOVES):
            if not grid: break
            
            pushable_crystals = list(grid.keys())
            if not pushable_crystals: break
            
            start_pos = self.np_random.choice(pushable_crystals, size=1)[0]
            start_pos = (start_pos[0], start_pos[1]) # Convert from numpy array if needed
            
            direction = self._get_direction_vector(self.np_random.integers(1, 5))

            # Simulate push without modifying self.grid
            chain = []
            current_pos = start_pos
            while current_pos in grid:
                chain.append(current_pos)
                current_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])
            
            end_pos = current_pos
            if 0 <= end_pos[0] < self.GRID_SIZE and 0 <= end_pos[1] < self.GRID_SIZE:
                new_scrambled_grid = grid.copy()
                for pos in reversed(chain):
                    crystal_type = new_scrambled_grid.pop(pos)
                    new_pos = (pos[0] + direction[0], pos[1] + direction[1])
                    new_scrambled_grid[new_pos] = crystal_type
                grid = new_scrambled_grid

        # Create a reverse map for quick lookups in reward calculation
        target_map = {pos: c_type for pos, c_type in target_grid.items()}

        return grid, target_grid, target_map

    def _calculate_target_distance(self, grid):
        total_dist = 0
        
        # Create a temporary copy of target positions to handle duplicate crystal types
        temp_target_positions = {}
        for pos, c_type in self.target_grid.items():
            if c_type not in temp_target_positions:
                temp_target_positions[c_type] = []
            temp_target_positions[c_type].append(pos)

        # For each crystal in the current grid, find its closest target of the same type
        for pos, c_type in grid.items():
            if c_type in temp_target_positions and temp_target_positions[c_type]:
                targets = temp_target_positions[c_type]
                distances = [abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in targets]
                min_dist = min(distances)
                
                # Find the target that produced the min_dist and remove it from consideration
                # to ensure one-to-one mapping for this step
                best_target_idx = distances.index(min_dist)
                temp_target_positions[c_type].pop(best_target_idx)
                
                total_dist += min_dist
            else:
                # Crystal that is not in the target solution, penalize heavily
                total_dist += self.GRID_SIZE * 2 
        return total_dist

    def _check_win(self):
        return self.grid == self.target_grid
    
    def _get_direction_vector(self, movement):
        if movement == 1: return (0, -1)  # Up
        if movement == 2: return (0, 1)   # Down
        if movement == 3: return (-1, 0)  # Left
        if movement == 4: return (1, 0)   # Right
        return (0, 0)

    def _create_particles(self, pos, color, count=5):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(0.05, 0.2)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': self.np_random.integers(10, 20),
                'max_life': 20,
                'color': color,
                'size': self.np_random.integers(2, 4)
            })

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1
        self.particles = [p for p in self.particles if p['life'] > 0]

    def close(self):
        pygame.quit()
        
    def validate_implementation(self):
        print("Running implementation validation...")
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
        assert trunc is False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")


if __name__ == "__main__":
    # This block allows you to play the game directly
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crystal Cavern")
    clock = pygame.time.Clock()
    
    running = True
    terminated = False
    
    while running:
        movement = 0 # No-op
        space_held = 0
        shift_held = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
                if event.key == pygame.K_ESCAPE:
                    running = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]: movement = 1
            elif keys[pygame.K_DOWN]: movement = 2
            elif keys[pygame.K_LEFT]: movement = 3
            elif keys[pygame.K_RIGHT]: movement = 4
            
            if keys[pygame.K_SPACE]: space_held = 1
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift_held = 1

            action = [movement, space_held, shift_held]
            obs, reward, terminated, truncated, info = env.step(action)
        
        # Display the observation from the environment
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit to 30 FPS for playability

    env.close()