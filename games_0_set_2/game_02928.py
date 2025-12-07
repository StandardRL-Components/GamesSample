
# Generated: 2025-08-27T21:51:26.718271
# Source Brief: brief_02928.md
# Brief Index: 2928

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import os
import pygame


# To ensure Pygame runs headlessly if no display is available
os.environ["SDL_VIDEODRIVER"] = "dummy"

class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    user_guide = (
        "Controls: Use arrow keys to move the crystal. Avoid the red traps and "
        "reach the golden exit before time runs out."
    )

    game_description = (
        "Navigate a crystal through trap-laden isometric caverns to reach the "
        "exit against the clock. Faster times yield higher scores."
    )

    auto_advance = False

    # --- Constants ---
    WIDTH, HEIGHT = 640, 400
    GRID_COLS, GRID_ROWS = 15, 15
    TILE_WIDTH, TILE_HEIGHT = 40, 20
    TILE_WIDTH_HALF, TILE_HEIGHT_HALF = TILE_WIDTH // 2, TILE_HEIGHT // 2
    MAX_STEPS = 600  # 60 seconds at 10 steps/sec

    # --- Colors ---
    COLOR_BG = (20, 25, 40)
    COLOR_TILE = (40, 50, 70)
    COLOR_TILE_BORDER = (60, 70, 90)
    COLOR_CRYSTAL = (0, 150, 255)
    COLOR_CRYSTAL_TOP = (100, 200, 255)
    COLOR_CRYSTAL_SHADOW = (0, 0, 0, 90)
    COLOR_TRAP = (255, 50, 50)
    COLOR_EXIT = (255, 200, 0)
    COLOR_TEXT = (220, 220, 240)
    COLOR_UI_BG = (30, 35, 50, 200)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        self.observation_space = Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.SysFont("monospace", 16, bold=True)
        self.font_game_over = pygame.font.SysFont("monospace", 48, bold=True)

        self.origin_x = self.WIDTH // 2
        self.origin_y = 80

        self.current_level = 1
        self.level_complete_in_last_episode = False
        
        self.reset()
        
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random = np.random.default_rng(seed)

        if self.level_complete_in_last_episode:
            self.current_level += 1
        self.level_complete_in_last_episode = False

        self._generate_level()

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_outcome = ""
        self.particles = []

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        self.steps += 1
        reward = -0.1  # Cost of existing per step

        # --- Movement Logic ---
        prev_pos = self.player_pos
        r, c = self.player_pos
        moves = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}  # Up, Down, Left, Right
        if movement in moves:
            dr, dc = moves[movement]
            new_r, new_c = r + dr, c + dc
            if 0 <= new_r < self.GRID_ROWS and 0 <= new_c < self.GRID_COLS:
                self.player_pos = (new_r, new_c)
                # Movement particles
                iso_x, iso_y = self._grid_to_iso(prev_pos[0], prev_pos[1])
                self._create_particles(
                    (iso_x, iso_y + self.TILE_HEIGHT_HALF),
                    self.COLOR_CRYSTAL, 3, 1, 10, (0.5, 0.5)
                )

        terminated = False

        # --- Collision and Termination Checks ---
        if self.player_pos in self.trap_locations:
            reward = -10
            self.score += reward
            terminated = True
            self.game_over = True
            self.game_outcome = "TRAPPED!"
            # Sound: sfx_trap_spring()
            iso_x, iso_y = self._grid_to_iso(self.player_pos[0], self.player_pos[1])
            self._create_particles((iso_x, iso_y), self.COLOR_TRAP, 30, 4, 20, (1, 1))

        elif self.player_pos == self.exit_pos:
            seconds_remaining = (self.MAX_STEPS - self.steps) / (self.MAX_STEPS / 60.0)
            time_bonus = math.floor(max(0, seconds_remaining)) * 10
            reward = 100 + time_bonus
            self.score += reward
            terminated = True
            self.game_over = True
            self.game_outcome = "VICTORY!"
            self.level_complete_in_last_episode = True
            # Sound: sfx_victory_fanfare()
            iso_x, iso_y = self._grid_to_iso(self.player_pos[0], self.player_pos[1])
            self._create_particles((iso_x, iso_y), self.COLOR_EXIT, 50, 5, 30, (1, 2))


        elif self.steps >= self.MAX_STEPS:
            # No extra reward penalty, the accumulation of -0.1 serves this purpose
            terminated = True
            self.game_over = True
            self.game_outcome = "TIME UP!"
            # Sound: sfx_timeout_buzzer()

        self.score = max(-100, min(9999, self.score + reward))

        return self._get_observation(), reward, terminated, False, self._get_info()

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_game()
        self._render_ui()
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _get_info(self):
        return {"score": self.score, "steps": self.steps, "level": self.current_level}
    
    # --- Level Generation ---
    def _generate_level(self):
        num_traps = min(3 + self.current_level - 1, (self.GRID_COLS * self.GRID_ROWS) // 4)
        
        while True:
            self.player_pos = (self.np_random.integers(0, 3), self.np_random.integers(0, 3))
            self.exit_pos = (self.GRID_ROWS - 1 - self.np_random.integers(0, 3), self.GRID_COLS - 1 - self.np_random.integers(0, 3))

            self.trap_locations = set()
            possible_locs = [(r, c) for r in range(self.GRID_ROWS) for c in range(self.GRID_COLS)]
            if self.player_pos in possible_locs: possible_locs.remove(self.player_pos)
            if self.exit_pos in possible_locs: possible_locs.remove(self.exit_pos)
            
            if len(possible_locs) >= num_traps:
                trap_indices = self.np_random.choice(len(possible_locs), num_traps, replace=False)
                for i in trap_indices:
                    self.trap_locations.add(possible_locs[i])
            
            if self._is_solvable():
                break

    def _is_solvable(self):
        q = deque([self.player_pos])
        visited = {self.player_pos}
        while q:
            r, c = q.popleft()
            if (r, c) == self.exit_pos:
                return True
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.GRID_ROWS and 0 <= nc < self.GRID_COLS and (nr, nc) not in visited and (nr, nc) not in self.trap_locations:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return False

    # --- Rendering Helpers ---
    def _render_game(self):
        # Back-to-front rendering for correct occlusion
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                iso_x, iso_y = self._grid_to_iso(r, c)
                
                self._draw_tile(iso_x, iso_y)

                if (r, c) in self.trap_locations:
                    self._draw_trap(iso_x, iso_y)
                elif (r, c) == self.exit_pos:
                    self._draw_exit(iso_x, iso_y)
                
                if (r, c) == self.player_pos and not self.game_over:
                    self._draw_player(iso_x, iso_y)

        self._update_and_draw_particles()

    def _render_ui(self):
        # UI Panel
        ui_surf = pygame.Surface((self.WIDTH, 40), pygame.SRCALPHA)
        ui_surf.fill(self.COLOR_UI_BG)
        self.screen.blit(ui_surf, (0, 0))
        pygame.draw.line(self.screen, self.COLOR_TILE_BORDER, (0, 40), (self.WIDTH, 40))

        # UI Text
        time_left = (self.MAX_STEPS - self.steps) / (self.MAX_STEPS / 60.0)
        score_text = self.font_ui.render(f"SCORE: {int(self.score):04d}", True, self.COLOR_TEXT)
        time_text = self.font_ui.render(f"TIME: {time_left:.1f}", True, self.COLOR_TEXT)
        level_text = self.font_ui.render(f"LEVEL: {self.current_level}", True, self.COLOR_TEXT)

        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (self.WIDTH - 150, 10))
        self.screen.blit(level_text, (self.WIDTH / 2 - level_text.get_width() / 2, 10))

        # Game Over Text
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            self.screen.blit(overlay, (0, 0))
            
            outcome_text = self.font_game_over.render(self.game_outcome, True, self.COLOR_EXIT if self.level_complete_in_last_episode else self.COLOR_TRAP)
            text_rect = outcome_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(outcome_text, text_rect)

    def _grid_to_iso(self, r, c):
        iso_x = self.origin_x + (c - r) * self.TILE_WIDTH_HALF
        iso_y = self.origin_y + (c + r) * self.TILE_HEIGHT_HALF
        return int(iso_x), int(iso_y)

    def _draw_tile(self, x, y):
        points = [
            (x, y),
            (x + self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),
            (x, y + self.TILE_HEIGHT),
            (x - self.TILE_WIDTH_HALF, y + self.TILE_HEIGHT_HALF),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_TILE)
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_TILE_BORDER)

    def _draw_trap(self, x, y):
        center_y = y + self.TILE_HEIGHT_HALF
        pygame.gfxdraw.filled_circle(self.screen, x, center_y, 8, self.COLOR_TRAP)
        pygame.gfxdraw.aacircle(self.screen, x, center_y, 8, self.COLOR_TRAP)

    def _draw_exit(self, x, y):
        center_y = y + self.TILE_HEIGHT_HALF
        # Glow effect
        for i in range(4, 0, -1):
            glow_color = (*self.COLOR_EXIT, 50)
            pygame.gfxdraw.filled_circle(self.screen, x, center_y, 8 + i * 3, glow_color)
        pygame.gfxdraw.filled_circle(self.screen, x, center_y, 8, self.COLOR_EXIT)
        pygame.gfxdraw.aacircle(self.screen, x, center_y, 8, self.COLOR_EXIT)

    def _draw_player(self, x, y):
        bob_y = math.sin(self.steps * 0.2) * 3
        center_y = y + self.TILE_HEIGHT_HALF + bob_y

        # Shadow
        shadow_rect = pygame.Rect(0, 0, self.TILE_WIDTH * 0.7, self.TILE_HEIGHT * 0.7)
        shadow_rect.center = (x, y + self.TILE_HEIGHT)
        shadow_surf = pygame.Surface(shadow_rect.size, pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, self.COLOR_CRYSTAL_SHADOW, (0, 0, *shadow_rect.size))
        self.screen.blit(shadow_surf, shadow_rect.topleft)

        # Crystal
        points_top = [
            (x, center_y - self.TILE_HEIGHT_HALF),
            (x + self.TILE_WIDTH_HALF * 0.8, center_y),
            (x, center_y + self.TILE_HEIGHT_HALF),
            (x - self.TILE_WIDTH_HALF * 0.8, center_y),
        ]
        points_bottom = [
            (x, center_y + self.TILE_HEIGHT_HALF * 2),
            (x + self.TILE_WIDTH_HALF * 0.8, center_y),
            (x, center_y + self.TILE_HEIGHT_HALF),
            (x - self.TILE_WIDTH_HALF * 0.8, center_y),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, points_bottom, self.COLOR_CRYSTAL)
        pygame.gfxdraw.filled_polygon(self.screen, points_top, self.COLOR_CRYSTAL_TOP)
        pygame.gfxdraw.aapolygon(self.screen, points_bottom, self.COLOR_CRYSTAL)
        pygame.gfxdraw.aapolygon(self.screen, points_top, self.COLOR_CRYSTAL_TOP)

    # --- Particle System ---
    def _create_particles(self, pos, color, count, speed, life, size_range):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            vel = self.np_random.uniform(0.5, 1.5) * speed
            self.particles.append({
                "pos": list(pos),
                "vel": [math.cos(angle) * vel, math.sin(angle) * vel],
                "life": self.np_random.integers(life // 2, life),
                "max_life": life,
                "color": color,
                "size": self.np_random.uniform(size_range[0], size_range[1]),
            })

    def _update_and_draw_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["life"] -= 1
            
            life_ratio = p["life"] / p["max_life"]
            current_size = int(p["size"] * life_ratio)
            if current_size > 0:
                color = (*p["color"], int(255 * life_ratio))
                pygame.gfxdraw.filled_circle(self.screen, int(p["pos"][0]), int(p["pos"][1]), current_size, color)
        
        self.particles = [p for p in self.particles if p["life"] > 0]
    
    def validate_implementation(self):
        """Call this at the end of __init__ to verify implementation."""
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
    # This block allows you to play the game with keyboard controls
    env = GameEnv()
    obs, info = env.reset()
    
    # Create a window to display the game
    pygame.display.set_caption("Crystal Caverns")
    display_screen = pygame.display.set_mode((GameEnv.WIDTH, GameEnv.HEIGHT))
    
    terminated = False
    running = True
    clock = pygame.time.Clock()

    while running:
        action = np.array([0, 0, 0])  # Default to no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                obs, info = env.reset()
                terminated = False

        if not terminated:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                action[0] = 1
            elif keys[pygame.K_DOWN]:
                action[0] = 2
            elif keys[pygame.K_LEFT]:
                action[0] = 3
            elif keys[pygame.K_RIGHT]:
                action[0] = 4
            
            # Since auto_advance is False, we only step when a key is pressed
            # or after a delay to allow for "no-op" steps.
            # For human play, we'll step on any key press.
            if action[0] != 0:
                obs, reward, terminated, truncated, info = env.step(action)
        
        # Get the observation from the environment
        frame = env._get_observation()
        # The observation is (H, W, C), but pygame surfaces expect (W, H).
        # We need to transpose it back.
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        
        # Draw the frame to the display window
        display_screen.blit(frame_surface, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit frame rate for human play

    pygame.quit()