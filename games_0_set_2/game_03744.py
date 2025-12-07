
# Generated: 2025-08-28T00:17:13.529613
# Source Brief: brief_03744.md
# Brief Index: 3744

        
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

    user_guide = (
        "Controls: Use arrow keys to move the cursor. Press Space to select a crystal, "
        "then move the cursor to an adjacent crystal and press Space again to swap them."
    )

    game_description = (
        "An isometric puzzle game. Swap adjacent crystals to form matches of three or more of the same color. "
        "Clear the entire board before you run out of moves or time!"
    )

    auto_advance = True
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Game constants
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_SIZE = 7
        self.NUM_COLORS = 3
        self.MAX_MOVES = 15
        self.TIME_LIMIT_SECONDS = 60
        self.FPS = 30
        self.MAX_STEPS = self.TIME_LIMIT_SECONDS * self.FPS

        # Visuals
        self.TILE_W, self.TILE_H = 60, 30
        self.ORIGIN_X = self.WIDTH // 2
        self.ORIGIN_Y = 80

        # Colors
        self.COLOR_BG = (25, 20, 35)
        self.COLOR_GRID = (60, 50, 80)
        self.CRYSTAL_COLORS = {
            1: (255, 50, 50),   # Red
            2: (50, 255, 50),   # Green
            3: (50, 100, 255),  # Blue
        }
        self.COLOR_UI_TEXT = (220, 220, 240)
        self.COLOR_CURSOR = (255, 255, 100)
        self.COLOR_SELECTED = (255, 150, 255)

        # Spaces
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        try:
            self.font_ui = pygame.font.SysFont("Consolas", 20, bold=True)
            self.font_msg = pygame.font.SysFont("Consolas", 32, bold=True)
        except pygame.error:
            self.font_ui = pygame.font.SysFont(None, 24)
            self.font_msg = pygame.font.SysFont(None, 40)

        # Game state is initialized in reset()
        self.grid = None
        self.cursor_pos = None
        self.selected_pos = None
        self.moves_left = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.win_message = None
        
        # Animation state
        self.game_phase = "IDLE" # IDLE, SWAP, MATCH, FALL
        self.animation_timer = 0
        self.animation_duration = 0
        self.animation_data = {}
        self.particles = []

        self.reset()
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.grid = self._generate_initial_board()
        self.cursor_pos = [self.GRID_SIZE // 2, self.GRID_SIZE // 2]
        self.selected_pos = None
        
        self.moves_left = self.MAX_MOVES
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win_message = ""
        
        self.game_phase = "IDLE"
        self.animation_timer = 0
        self.animation_data = {}
        self.particles = []
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.auto_advance:
            self.clock.tick(self.FPS)

        reward = 0
        terminated = False
        
        self.steps += 1
        
        if not self.game_over:
            if self.game_phase == "IDLE":
                reward += self._handle_input(action)
            else:
                self._update_animations()

        # Check for termination conditions
        if not self.game_over:
            if self.moves_left <= 0:
                self.game_over = True
                self.win_message = "Out of Moves!"
                reward += -100
            elif self.steps >= self.MAX_STEPS:
                self.game_over = True
                self.win_message = "Time's Up!"
                reward += -100
            elif np.count_nonzero(self.grid) == 0:
                self.game_over = True
                self.win_message = "Board Cleared!"
                reward += 100
        
        terminated = self.game_over

        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _handle_input(self, action):
        movement, space_pressed, _ = action
        space_pressed = space_pressed == 1
        reward = 0

        # --- Cursor Movement ---
        if movement == 1: # Up
            self.cursor_pos[0] = max(0, self.cursor_pos[0] - 1)
        elif movement == 2: # Down
            self.cursor_pos[0] = min(self.GRID_SIZE - 1, self.cursor_pos[0] + 1)
        elif movement == 3: # Left
            self.cursor_pos[1] = max(0, self.cursor_pos[1] - 1)
        elif movement == 4: # Right
            self.cursor_pos[1] = min(self.GRID_SIZE - 1, self.cursor_pos[1] + 1)

        # --- Selection / Swap ---
        if space_pressed:
            if self.selected_pos is None:
                # Select a crystal if it exists
                if self.grid[self.cursor_pos[0], self.cursor_pos[1]] > 0:
                    self.selected_pos = list(self.cursor_pos)
                    # sfx: select_crystal.wav
            else:
                # Attempt a swap
                is_adjacent = abs(self.selected_pos[0] - self.cursor_pos[0]) + abs(self.selected_pos[1] - self.cursor_pos[1]) == 1
                is_on_self = self.selected_pos == self.cursor_pos
                
                if is_on_self:
                    self.selected_pos = None # Deselect
                elif is_adjacent:
                    self.moves_left -= 1
                    reward = -0.2 # Penalty for making a move
                    
                    # Start swap animation
                    self._start_animation("SWAP", 0.2, {
                        "pos1": self.selected_pos,
                        "pos2": self.cursor_pos
                    })
                    # sfx: swap.wav
                    self.selected_pos = None
                else:
                    # Invalid swap, just change selection
                    if self.grid[self.cursor_pos[0], self.cursor_pos[1]] > 0:
                        self.selected_pos = list(self.cursor_pos)
                        # sfx: select_crystal.wav
                    else:
                        self.selected_pos = None # Deselect if moving to empty space

        return reward

    def _update_animations(self):
        self.animation_timer += 1 / self.FPS
        progress = min(1.0, self.animation_timer / self.animation_duration)

        if progress >= 1.0:
            # --- Finish SWAP ---
            if self.game_phase == "SWAP":
                pos1, pos2 = self.animation_data["pos1"], self.animation_data["pos2"]
                self.grid[pos1[0], pos1[1]], self.grid[pos2[0], pos2[1]] = self.grid[pos2[0], pos2[1]], self.grid[pos1[0], pos1[1]]
                
                matches = self._find_matches()
                if matches:
                    self._start_animation("MATCH", 0.3, {"matches": matches})
                else: # Invalid move, swap back
                    self._start_animation("SWAP_BACK", 0.2, {"pos1": pos1, "pos2": pos2})
                    # sfx: invalid_move.wav
            
            # --- Finish SWAP_BACK ---
            elif self.game_phase == "SWAP_BACK":
                pos1, pos2 = self.animation_data["pos1"], self.animation_data["pos2"]
                self.grid[pos1[0], pos1[1]], self.grid[pos2[0], pos2[1]] = self.grid[pos2[0], pos2[1]], self.grid[pos1[0], pos1[1]]
                self.game_phase = "IDLE"

            # --- Finish MATCH ---
            elif self.game_phase == "MATCH":
                matches = self.animation_data["matches"]
                num_cleared = len(matches)
                
                # Calculate reward for match
                self.score += num_cleared # +1 per crystal
                if num_cleared >= 4: self.score += 5 # Bonus
                
                for r, c in matches:
                    self._create_particles(r, c)
                    self.grid[r, c] = 0
                # sfx: match_clear.wav
                
                fall_data = self._get_fall_data()
                if fall_data:
                    self._start_animation("FALL", 0.25, {"falls": fall_data})
                else:
                    self.game_phase = "IDLE"

            # --- Finish FALL ---
            elif self.game_phase == "FALL":
                # Apply the gravity change to the grid
                temp_grid = np.zeros_like(self.grid)
                for r_from, c_from, r_to, c_to in self.animation_data["falls"]:
                    temp_grid[r_to, c_to] = self.grid[r_from, c_from]
                
                # Copy non-falling crystals
                for r in range(self.GRID_SIZE):
                    for c in range(self.GRID_SIZE):
                        is_falling = any(r == f[0] and c == f[1] for f in self.animation_data["falls"])
                        if not is_falling and self.grid[r, c] > 0:
                            temp_grid[r, c] = self.grid[r, c]
                self.grid = temp_grid

                # Check for chain reactions
                matches = self._find_matches()
                if matches:
                    self._start_animation("MATCH", 0.3, {"matches": matches})
                    # sfx: chain_reaction.wav
                else:
                    self.game_phase = "IDLE"

    def _start_animation(self, phase, duration, data):
        self.game_phase = phase
        self.animation_duration = duration
        self.animation_timer = 0
        self.animation_data = data

    def _get_fall_data(self):
        fall_data = []
        for c in range(self.GRID_SIZE):
            empty_count = 0
            for r in range(self.GRID_SIZE - 1, -1, -1):
                if self.grid[r, c] == 0:
                    empty_count += 1
                elif empty_count > 0:
                    fall_data.append((r, c, r + empty_count, c))
        return fall_data

    def _find_matches(self):
        matches = set()
        # Horizontal
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE - 2):
                color = self.grid[r, c]
                if color > 0 and color == self.grid[r, c+1] and color == self.grid[r, c+2]:
                    matches.update([(r, c), (r, c+1), (r, c+2)])
        # Vertical
        for c in range(self.GRID_SIZE):
            for r in range(self.GRID_SIZE - 2):
                color = self.grid[r, c]
                if color > 0 and color == self.grid[r+1, c] and color == self.grid[r+2, c]:
                    matches.update([(r, c), (r+1, c), (r+2, c)])
        return list(matches)

    def _generate_initial_board(self):
        while True:
            grid = self.np_random.integers(1, self.NUM_COLORS + 1, size=(self.GRID_SIZE, self.GRID_SIZE))
            
            # Temporarily set grid to check for matches
            original_grid = self.grid
            self.grid = grid
            if not self._find_matches():
                self.grid = original_grid
                return grid
            self.grid = original_grid

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
            "moves_left": self.moves_left,
            "time_left": (self.MAX_STEPS - self.steps) / self.FPS
        }

    def _render_game(self):
        self._draw_grid()
        self._update_and_draw_particles()

        # --- Draw Crystals ---
        drawn_in_animation = set()
        
        # Draw animated crystals first
        if self.game_phase in ["SWAP", "SWAP_BACK", "FALL"]:
            progress = self.animation_timer / self.animation_duration
            
            if self.game_phase in ["SWAP", "SWAP_BACK"]:
                pos1, pos2 = self.animation_data["pos1"], self.animation_data["pos2"]
                c1_r, c1_c = pos1
                c2_r, c2_c = pos2
                
                interp_r1 = c1_r + (c2_r - c1_r) * progress
                interp_c1 = c1_c + (c2_c - c1_c) * progress
                interp_r2 = c2_r + (c1_r - c2_r) * progress
                interp_c2 = c2_c + (c1_c - c2_c) * progress

                color1 = self.grid[c1_r, c1_c]
                color2 = self.grid[c2_r, c2_c]
                if self.game_phase == "SWAP_BACK":
                    color1, color2 = color2, color1

                self._draw_crystal(interp_r1, interp_c1, color1)
                self._draw_crystal(interp_r2, interp_c2, color2)
                drawn_in_animation.add(tuple(pos1))
                drawn_in_animation.add(tuple(pos2))

            elif self.game_phase == "FALL":
                for r_from, c_from, r_to, c_to in self.animation_data["falls"]:
                    interp_r = r_from + (r_to - r_from) * progress
                    interp_c = c_from + (c_to - c_from) * progress
                    color = self.grid[r_from, c_from]
                    self._draw_crystal(interp_r, interp_c, color)
                    drawn_in_animation.add((r_from, c_from))

        # Draw static crystals
        for r in range(self.GRID_SIZE):
            for c in range(self.GRID_SIZE):
                if (r,c) in drawn_in_animation: continue
                
                color_idx = self.grid[r, c]
                if color_idx > 0:
                    scale = 1.0
                    if self.game_phase == "MATCH":
                        if (r, c) in self.animation_data["matches"]:
                            progress = self.animation_timer / self.animation_duration
                            scale = 1.0 + math.sin(progress * math.pi) * 0.5 # pulse
                    
                    self._draw_crystal(r, c, color_idx, scale)

        # --- Draw Cursor and Selection ---
        if not self.game_over:
            # Draw selected highlight
            if self.selected_pos is not None:
                self._draw_iso_poly(self.selected_pos[0], self.selected_pos[1], self.COLOR_SELECTED, 1.3, filled=False, width=3)
            # Draw cursor
            self._draw_iso_poly(self.cursor_pos[0], self.cursor_pos[1], self.COLOR_CURSOR, 1.2, filled=False, width=3)

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH // 2 - score_text.get_width() // 2, self.HEIGHT - 30))

        # Time
        time_left = max(0, (self.MAX_STEPS - self.steps) / self.FPS)
        time_text = self.font_ui.render(f"Time: {time_left:.1f}", True, self.COLOR_UI_TEXT)
        self.screen.blit(time_text, (10, 10))

        # Moves
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(moves_text, (self.WIDTH - moves_text.get_width() - 10, 10))
        
        # Game Over Message
        if self.game_over:
            msg_surf = self.font_msg.render(self.win_message, True, self.COLOR_UI_TEXT)
            self.screen.blit(msg_surf, (self.WIDTH // 2 - msg_surf.get_width() // 2, self.HEIGHT // 2 - msg_surf.get_height() // 2))

    def _iso_to_screen(self, r, c):
        x = self.ORIGIN_X + (c - r) * self.TILE_W / 2
        y = self.ORIGIN_Y + (c + r) * self.TILE_H / 2
        return int(x), int(y)

    def _draw_grid(self):
        for r in range(self.GRID_SIZE + 1):
            p1 = self._iso_to_screen(r, 0)
            p2 = self._iso_to_screen(r, self.GRID_SIZE)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)
        for c in range(self.GRID_SIZE + 1):
            p1 = self._iso_to_screen(0, c)
            p2 = self._iso_to_screen(self.GRID_SIZE, c)
            pygame.draw.line(self.screen, self.COLOR_GRID, p1, p2, 1)

    def _draw_iso_poly(self, r, c, color, scale=1.0, filled=True, width=0):
        center_x, center_y = self._iso_to_screen(r, c)
        w = (self.TILE_W / 2) * scale
        h = (self.TILE_H / 2) * scale
        points = [
            (center_x, center_y - h),
            (center_x + w, center_y),
            (center_x, center_y + h),
            (center_x - w, center_y),
        ]
        if filled:
            pygame.gfxdraw.filled_polygon(self.screen, points, color)
        else:
            pygame.gfxdraw.aapolygon(self.screen, points, color)
            if width > 1: # Thicken line
                 for i in range(len(points)):
                    pygame.draw.line(self.screen, color, points[i], points[(i + 1) % len(points)], width)

    def _draw_crystal(self, r, c, color_idx, scale=1.0):
        if color_idx == 0: return

        base_color = self.CRYSTAL_COLORS[color_idx]
        highlight_color = tuple(min(255, val + 80) for val in base_color)
        shadow_color = tuple(max(0, val - 80) for val in base_color)

        center_x, center_y = self._iso_to_screen(r, c)
        w = (self.TILE_W / 2) * scale
        h = (self.TILE_H / 2) * scale
        depth = self.TILE_H * 0.6 * scale

        top_face = [
            (center_x, center_y - h), (center_x + w, center_y),
            (center_x, center_y + h), (center_x - w, center_y)
        ]
        left_face = [
            (center_x - w, center_y), (center_x, center_y + h),
            (center_x, center_y + h + depth), (center_x - w, center_y + depth)
        ]
        right_face = [
            (center_x + w, center_y), (center_x, center_y + h),
            (center_x, center_y + h + depth), (center_x + w, center_y + depth)
        ]

        # Glow effect
        glow_radius = int(w * 1.2)
        glow_surf = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*base_color, 60), (glow_radius, glow_radius), glow_radius)
        self.screen.blit(glow_surf, (center_x - glow_radius, center_y - glow_radius + depth/2), special_flags=pygame.BLEND_RGBA_ADD)

        pygame.gfxdraw.filled_polygon(self.screen, left_face, shadow_color)
        pygame.gfxdraw.filled_polygon(self.screen, right_face, shadow_color)
        pygame.gfxdraw.filled_polygon(self.screen, top_face, base_color)
        pygame.gfxdraw.aapolygon(self.screen, top_face, highlight_color)
        
        # Top highlight
        hw = w * 0.5
        hh = h * 0.5
        highlight_points = [
            (center_x, center_y - hh), (center_x + hw, center_y),
            (center_x, center_y + hh), (center_x - hw, center_y)
        ]
        pygame.gfxdraw.filled_polygon(self.screen, highlight_points, highlight_color)

    def _create_particles(self, r, c):
        x, y = self._iso_to_screen(r, c)
        color = self.CRYSTAL_COLORS[self.grid[r,c]]
        for _ in range(15):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': [x, y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 1],
                'life': random.uniform(0.5, 1.0),
                'color': color
            })

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['life'] -= 1 / self.FPS
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['life'] / 1.0))
                color = (*p['color'], alpha)
                pygame.draw.circle(self.screen, color, p['pos'], int(p['life'] * 3 + 1))


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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # --- Pygame setup for human play ---
    screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crystal Cavern")
    clock = pygame.time.Clock()
    
    running = True
    total_reward = 0
    
    while running:
        movement, space, shift = 0, 0, 0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                total_reward = 0

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: movement = 1
        elif keys[pygame.K_DOWN]: movement = 2
        elif keys[pygame.K_LEFT]: movement = 3
        elif keys[pygame.K_RIGHT]: movement = 4
        
        if keys[pygame.K_SPACE]: space = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1

        action = [movement, space, shift]
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Render the observation to the display
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

        if terminated:
            print(f"Game Over! Final Score: {info['score']}, Total Reward: {total_reward}")
            # Wait for a moment before allowing reset
            pygame.time.wait(2000)
            obs, info = env.reset()
            total_reward = 0

        clock.tick(env.FPS)
        
    pygame.quit()