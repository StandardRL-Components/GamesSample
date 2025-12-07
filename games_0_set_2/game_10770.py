import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T16:01:42.055422
# Source Brief: brief_00770.md
# Brief Index: 770
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque

class GameEnv(gym.Env):
    """
    Crystal Growth Environment for Gymnasium.

    The player places mineral cards on a hexagonal grid and selects growth patterns
    to cultivate crystals. The goal is to maximize the final purity score by
    triggering large chain reactions of same-colored minerals.
    """
    metadata = {"render_modes": ["rgb_array"]}

    game_description = (
        "Strategically place minerals on a hexagonal grid to grow crystals. Trigger chain reactions and "
        "select growth patterns to maximize your final purity score."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move the cursor. Press space to place a mineral and "
        "shift to cycle through available growth patterns."
    )
    auto_advance = True

    # --- CONSTANTS ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GAME_AREA_WIDTH = 480
    UI_WIDTH = SCREEN_WIDTH - GAME_AREA_WIDTH

    GRID_COLS = 17
    GRID_ROWS = 13
    CELL_RADIUS = 14

    MAX_STEPS = 500 # A full game is 500 placements

    # --- COLORS ---
    COLOR_BG = (15, 18, 28)
    COLOR_GRID = (40, 45, 60)
    COLOR_CURSOR = (255, 255, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (25, 30, 45)
    COLOR_UI_HEADER = (60, 70, 90)

    MINERAL_COLORS = {
        1: (255, 80, 80),   # Red
        2: (80, 120, 255),  # Blue
        3: (80, 255, 120),  # Green
        4: (255, 120, 255), # Magenta
        5: (255, 200, 80),  # Orange
    }
    COLOR_PURE = (255, 255, 255)

    # --- GROWTH PATTERNS (Axial Hex Coordinates) ---
    PATTERNS = {
        "Triad": [(1, 0), (0, -1), (-1, 1)],
        "Line": [(-1, 0), (1, 0)],
        "Cluster": [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)],
        "Arrow": [(1, 0), (2, 0), (0, -1), (-1, 1)],
        "Fork": [(1, -1), (-1, 1), (0, 1), (0, -1)],
    }
    HEX_DIRECTIONS = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]


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
        self.font_sm = pygame.font.Font(None, 22)
        self.font_md = pygame.font.Font(None, 28)
        self.font_lg = pygame.font.Font(None, 36)

        # State variables initialized in reset()
        self.grid = None
        self.cursor_q, self.cursor_r = 0, 0
        self.visual_cursor_pos = [0, 0]
        self.steps = 0
        self.score = 0.0
        self.purity = 0.0
        self.game_over = False
        self.current_mineral_card = 0
        self.available_minerals = []
        self.unlocked_patterns = []
        self.selected_pattern_idx = 0
        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

        self.reset()
        # self.validate_implementation() # Commented out for production

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.grid = [[None for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self.cursor_q, self.cursor_r = self.GRID_COLS // 2, self.GRID_ROWS // 2
        
        center_pixel_pos = self._hex_to_pixel(self.cursor_q, self.cursor_r)
        self.visual_cursor_pos = [float(center_pixel_pos[0]), float(center_pixel_pos[1])]

        self.steps = 0
        self.score = 0.0
        self.purity = 0.0
        self.game_over = False
        
        self.available_minerals = [1, 2]
        self.unlocked_patterns = ["Triad", "Line"]
        self.selected_pattern_idx = 0
        self._draw_new_card()

        self.particles = []
        self.prev_space_held = False
        self.prev_shift_held = False

        # Place a seed crystal to start
        self._place_crystal(self.cursor_q, self.cursor_r, self.current_mineral_card)
        self._draw_new_card()

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement, space_val, shift_val = action
        space_held = space_val == 1
        shift_held = shift_val == 1
        reward = 0.0

        # --- Handle Actions (Edge-triggered for discrete actions) ---
        
        # 1. Cycle growth pattern (on Shift press)
        if shift_held and not self.prev_shift_held:
            self.selected_pattern_idx = (self.selected_pattern_idx + 1) % len(self.unlocked_patterns)
            # sfx: UI_CLICK

        # 2. Move cursor
        if movement != 0:
            dq, dr = self._get_move_from_action(movement)
            new_q, new_r = self.cursor_q + dq, self.cursor_r + dr
            if 0 <= new_r < self.GRID_ROWS and 0 <= new_q < self.GRID_COLS:
                self.cursor_q, self.cursor_r = new_q, new_r

        # 3. Place mineral (on Space press)
        if space_held and not self.prev_space_held:
            placed, growth_count, chain_count = self._attempt_placement()
            if placed:
                reward += growth_count * 0.1  # Scaled reward
                reward += chain_count * 0.5   # Scaled reward
                self.steps += 1
                self._draw_new_card()
                # sfx: PLACE_CRYSTAL, CHAIN_REACTION

        self.prev_space_held = space_held
        self.prev_shift_held = shift_held

        # --- Update Game State ---
        self._update_game_logic()
        self._update_particles()
        self._update_animations()

        terminated = self.steps >= self.MAX_STEPS
        truncated = False
        if terminated and not self.game_over:
            self.purity = self._calculate_final_purity()
            terminal_reward = self.purity * 50.0
            reward += terminal_reward
            self.game_over = True
            
        self.score += reward

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _update_game_logic(self):
        # Introduce new minerals/patterns over time
        if self.steps > 0 and self.steps % 50 == 0:
            if self.steps == 50 and "Cluster" not in self.unlocked_patterns:
                self.unlocked_patterns.append("Cluster")
            if self.steps == 100 and 3 not in self.available_minerals:
                self.available_minerals.append(3)
            if self.steps == 150 and "Arrow" not in self.unlocked_patterns:
                self.unlocked_patterns.append("Arrow")
            if self.steps == 200 and 4 not in self.available_minerals:
                self.available_minerals.append(4)
            if self.steps == 250 and "Fork" not in self.unlocked_patterns:
                self.unlocked_patterns.append("Fork")
            if self.steps == 300 and 5 not in self.available_minerals:
                self.available_minerals.append(5)


    def _get_move_from_action(self, movement):
        # Map cardinal directions to nearest hex directions
        # This mapping depends on the row's parity for up/down moves
        is_even_row = self.cursor_r % 2 == 0
        if movement == 1: return (0, -1) # Up
        if movement == 2: return (0, 1)  # Down
        if movement == 3: # Left
            return (-1, 0)
        if movement == 4: # Right
            return (1, 0)
        return (0, 0)

    def _attempt_placement(self):
        if self.grid[self.cursor_r][self.cursor_q] is None:
            self._place_crystal(self.cursor_q, self.cursor_r, self.current_mineral_card)
            growth_count, chain_count = self._propagate_growth(self.cursor_q, self.cursor_r)
            return True, growth_count, chain_count
        return False, 0, 0

    def _place_crystal(self, q, r, mineral_type):
        purity = self._calculate_local_purity(q, r, mineral_type)
        self.grid[r][q] = Crystal(mineral_type, (q, r), purity)

    def _propagate_growth(self, start_q, start_r):
        queue = deque([(start_q, start_r)])
        processed_this_turn = set([(start_q, start_r)])
        growth_count, chain_count = 0, 0
        
        pattern_name = self.unlocked_patterns[self.selected_pattern_idx]
        pattern_coords = self.PATTERNS[pattern_name]

        while queue:
            q, r = queue.popleft()
            crystal = self.grid[r][q]
            if not crystal: continue
            
            # Growth from pattern
            for dq, dr in pattern_coords:
                nq, nr = q + dq, r + dr
                if 0 <= nr < self.GRID_ROWS and 0 <= nq < self.GRID_COLS and \
                   self.grid[nr][nq] is None and (nq, nr) not in processed_this_turn:
                    self._place_crystal(nq, nr, crystal.type)
                    growth_count += 1
                    processed_this_turn.add((nq, nr))
                    queue.append((nq, nr))
                    self._spawn_particles(self._hex_to_pixel(nq, nr), crystal.color, 5, 1.5)

            # Chain reaction to neighbors
            for dq, dr in self.HEX_DIRECTIONS:
                nq, nr = q + dq, r + dr
                if 0 <= nr < self.GRID_ROWS and 0 <= nq < self.GRID_COLS and \
                   (nq, nr) not in processed_this_turn:
                    neighbor = self.grid[nr][nq]
                    if neighbor and neighbor.type == crystal.type:
                        chain_count += 1
                        processed_this_turn.add((nq, nr))
                        queue.append((nq, nr))
                        self._spawn_particles(self._hex_to_pixel(nq, nr), crystal.color, 10, 2.5)
        return growth_count, chain_count
    
    def _calculate_local_purity(self, q, r, mineral_type):
        same_type_neighbors = 0
        total_neighbors = 0
        for dq, dr in self.HEX_DIRECTIONS:
            nq, nr = q + dq, r + dr
            if 0 <= nr < self.GRID_ROWS and 0 <= nq < self.GRID_COLS:
                neighbor = self.grid[nr][nq]
                if neighbor:
                    total_neighbors += 1
                    if neighbor.type == mineral_type:
                        same_type_neighbors += 1
        return (same_type_neighbors + 1) / (total_neighbors + 1) # Add self to numerator and denominator

    def _calculate_final_purity(self):
        total_purity = 0
        crystal_count = 0
        for r in range(self.GRID_ROWS):
            for q in range(self.GRID_COLS):
                if self.grid[r][q]:
                    total_purity += self.grid[r][q].purity
                    crystal_count += 1
        return total_purity / crystal_count if crystal_count > 0 else 0

    def _draw_new_card(self):
        if self.available_minerals:
            self.current_mineral_card = self.np_random.choice(self.available_minerals)

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "purity": self.purity}

    def _render_game(self):
        self._draw_grid()
        self._draw_crystals()
        self._draw_particles()
        self._draw_cursor()

    def _render_ui(self):
        ui_rect = pygame.Rect(self.GAME_AREA_WIDTH, 0, self.UI_WIDTH, self.SCREEN_HEIGHT)
        pygame.draw.rect(self.screen, self.COLOR_UI_BG, ui_rect)
        pygame.draw.line(self.screen, self.COLOR_UI_HEADER, (self.GAME_AREA_WIDTH, 0), (self.GAME_AREA_WIDTH, self.SCREEN_HEIGHT), 2)

        y_offset = 15

        # Score
        score_text = self.font_lg.render(f"{self.score:.1f}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (self.GAME_AREA_WIDTH + 20, y_offset))
        y_offset += 35
        score_label = self.font_sm.render("SCORE", True, self.COLOR_UI_HEADER)
        self.screen.blit(score_label, (self.GAME_AREA_WIDTH + 20, y_offset))
        y_offset += 40

        # Turn
        turn_text = self.font_lg.render(f"{self.steps}/{self.MAX_STEPS}", True, self.COLOR_TEXT)
        self.screen.blit(turn_text, (self.GAME_AREA_WIDTH + 20, y_offset))
        y_offset += 35
        turn_label = self.font_sm.render("TURN", True, self.COLOR_UI_HEADER)
        self.screen.blit(turn_label, (self.GAME_AREA_WIDTH + 20, y_offset))
        y_offset += 50

        # Current Card
        card_label = self.font_sm.render("CURRENT MINERAL", True, self.COLOR_UI_HEADER)
        self.screen.blit(card_label, (self.GAME_AREA_WIDTH + 20, y_offset))
        y_offset += 40
        card_color = self.MINERAL_COLORS.get(self.current_mineral_card, (50, 50, 50))
        self._draw_hexagon(self.screen, card_color, (self.GAME_AREA_WIDTH + self.UI_WIDTH // 2, y_offset), self.CELL_RADIUS * 1.5, is_ui=True)
        y_offset += 60

        # Selected Pattern
        pattern_label = self.font_sm.render("GROWTH PATTERN", True, self.COLOR_UI_HEADER)
        self.screen.blit(pattern_label, (self.GAME_AREA_WIDTH + 20, y_offset))
        y_offset += 25
        pattern_name = self.unlocked_patterns[self.selected_pattern_idx]
        pattern_text = self.font_md.render(pattern_name, True, self.COLOR_TEXT)
        self.screen.blit(pattern_text, (self.GAME_AREA_WIDTH + 20, y_offset))
        
        if self.game_over:
            overlay = pygame.Surface((self.GAME_AREA_WIDTH, 100), pygame.SRCALPHA)
            overlay.fill((15, 18, 28, 220))
            self.screen.blit(overlay, (0, self.SCREEN_HEIGHT // 2 - 50))
            
            end_text = self.font_lg.render("GAME OVER", True, self.COLOR_TEXT)
            text_rect = end_text.get_rect(center=(self.GAME_AREA_WIDTH // 2, self.SCREEN_HEIGHT // 2 - 20))
            self.screen.blit(end_text, text_rect)
            
            purity_text = self.font_md.render(f"Final Purity: {self.purity:.2%}", True, self.COLOR_TEXT)
            purity_rect = purity_text.get_rect(center=(self.GAME_AREA_WIDTH // 2, self.SCREEN_HEIGHT // 2 + 20))
            self.screen.blit(purity_text, purity_rect)


    def _draw_grid(self):
        for r in range(self.GRID_ROWS):
            for q in range(self.GRID_COLS):
                center_pos = self._hex_to_pixel(q, r)
                self._draw_hexagon(self.screen, self.COLOR_GRID, center_pos, self.CELL_RADIUS, 1)

    def _draw_crystals(self):
        for r in range(self.GRID_ROWS):
            for q in range(self.GRID_COLS):
                crystal = self.grid[r][q]
                if crystal:
                    crystal.draw(self.screen, self)
    
    def _draw_cursor(self):
        target_pos = self._hex_to_pixel(self.cursor_q, self.cursor_r)
        # Interpolate for smooth movement
        self.visual_cursor_pos[0] += (target_pos[0] - self.visual_cursor_pos[0]) * 0.4
        self.visual_cursor_pos[1] += (target_pos[1] - self.visual_cursor_pos[1]) * 0.4
        
        self._draw_hexagon(self.screen, self.COLOR_CURSOR, self.visual_cursor_pos, self.CELL_RADIUS + 2, 2)
        
    def _hex_to_pixel(self, q, r):
        # Axial to pixel for pointy-top hex
        x = self.CELL_RADIUS * (math.sqrt(3) * q + math.sqrt(3) / 2 * r)
        y = self.CELL_RADIUS * (3. / 2 * r)
        return int(x + 30), int(y + 30)

    def _draw_hexagon(self, surface, color, center, radius, width=0, is_ui=False):
        points = []
        for i in range(6):
            angle_deg = 60 * i
            angle_rad = math.pi / 180 * angle_deg
            points.append((center[0] + radius * math.cos(angle_rad),
                           center[1] + radius * math.sin(angle_rad)))
        
        # Glow effect for game crystals
        if width == 0 and not is_ui:
            glow_radius = radius * 1.8
            glow_points = []
            for i in range(6):
                angle_deg = 60 * i
                angle_rad = math.pi / 180 * angle_deg
                glow_points.append((center[0] + glow_radius * math.cos(angle_rad),
                                    center[1] + glow_radius * math.sin(angle_rad)))
            
            # Use gfxdraw for anti-aliased, filled polygon
            glow_color = (*color, 30) # Add alpha
            pygame.gfxdraw.aapolygon(surface, glow_points, glow_color)
            pygame.gfxdraw.filled_polygon(surface, glow_points, glow_color)

        pygame.draw.polygon(surface, color, points, width)

    def _spawn_particles(self, pos, color, count, speed_mult):
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1.0, 2.5) * speed_mult
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = random.uniform(0.3, 0.8)
            self.particles.append({'pos': list(pos), 'vel': vel, 'life': lifetime, 'max_life': lifetime, 'color': color})

    def _update_particles(self):
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][0] *= 0.95 # friction
            p['vel'][1] *= 0.95
            p['life'] -= 1/30.0 # Assuming 30fps
        self.particles = [p for p in self.particles if p['life'] > 0]

    def _draw_particles(self):
        for p in self.particles:
            life_ratio = max(0, p['life'] / p['max_life'])
            radius = int(life_ratio * 3)
            if radius > 0:
                color = tuple(int(c * life_ratio) for c in p['color'])
                pygame.draw.circle(self.screen, color, [int(p['pos'][0]), int(p['pos'][1])], radius)

    def _update_animations(self):
        for r in range(self.GRID_ROWS):
            for q in range(self.GRID_COLS):
                if self.grid[r][q]:
                    self.grid[r][q].update()

    def validate_implementation(self):
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        print("✓ Implementation validated successfully")

class Crystal:
    def __init__(self, mineral_type, pos_hex, purity):
        self.type = mineral_type
        self.pos_hex = pos_hex
        self.purity = purity
        self.color = GameEnv.MINERAL_COLORS.get(mineral_type, (255, 255, 255))
        self.scale = 0.0 # For spawn animation
        self.pulse = 0.0
        self.pulse_speed = random.uniform(0.05, 0.1)

    def update(self):
        self.scale = min(1.0, self.scale + 0.1)
        self.pulse = (self.pulse + self.pulse_speed) % (2 * math.pi)

    def draw(self, surface, env):
        if self.scale <= 0: return
        
        center_pos = env._hex_to_pixel(self.pos_hex[0], self.pos_hex[1])
        base_radius = env.CELL_RADIUS * self.scale
        
        # Draw main crystal body
        pulse_effect = 1 + 0.05 * math.sin(self.pulse)
        radius = base_radius * pulse_effect
        env._draw_hexagon(surface, self.color, center_pos, radius)

        # Draw purity core
        purity_radius = radius * self.purity
        if purity_radius > 1:
            core_color = tuple(min(255, c + 80) for c in self.color)
            mixed_color = tuple(int(c1 * (1-self.purity) + c2 * self.purity) for c1, c2 in zip(self.color, GameEnv.COLOR_PURE))
            env._draw_hexagon(surface, mixed_color, center_pos, purity_radius)

if __name__ == '__main__':
    # This block is for human play and is not part of the Gymnasium environment API.
    # It will not be executed by the test suite.
    # To use, you must have pygame installed and not be in a headless environment.
    try:
        # Re-initialize pygame with video
        os.environ.pop("SDL_VIDEODRIVER", None)
        pygame.quit() # Quit the dummy instance
        pygame.init()
        pygame.font.init()

        env = GameEnv(render_mode="rgb_array")
        obs, info = env.reset()
        
        screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
        pygame.display.set_caption("Crystal Growth")
        clock = pygame.time.Clock()
        
        running = True
        terminated = False
        
        while running:
            movement = 0 # No-op
            space = 0
            shift = 0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            if not terminated:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_UP]: movement = 1
                elif keys[pygame.K_DOWN]: movement = 2
                elif keys[pygame.K_LEFT]: movement = 3
                elif keys[pygame.K_RIGHT]: movement = 4
                
                if keys[pygame.K_SPACE]: space = 1
                if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]: shift = 1
                
                action = [movement, space, shift]
                obs, reward, terminated, truncated, info = env.step(action)

            # Render the observation from the environment
            surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
            screen.blit(surf, (0, 0))
            pygame.display.flip()
            
            clock.tick(30) # Limit frame rate

        pygame.quit()
    except pygame.error as e:
        print(f"Pygame error: {e}")
        print("Could not run human-playable demo. This is expected in a headless environment.")