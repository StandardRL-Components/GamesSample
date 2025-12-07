
# Generated: 2025-08-27T21:30:40.401725
# Source Brief: brief_02810.md
# Brief Index: 2810

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to move the crystal on the grid."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "An isometric puzzle game. Move the crystal onto tiles that match its color "
        "to fill the target slots. Fill all 10 slots within 3 moves to win."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # Screen and grid dimensions
        self.WIDTH, self.HEIGHT = 640, 400
        self.GRID_W, self.GRID_H = 10, 10
        self.TILE_W, self.TILE_H = 48, 24
        self.TILE_W_HALF, self.TILE_H_HALF = self.TILE_W // 2, self.TILE_H // 2
        
        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.HEIGHT, self.WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        
        # Visuals
        self.font_main = pygame.font.SysFont("Consolas", 20, bold=True)
        self.font_small = pygame.font.SysFont("Consolas", 14)

        self.COLOR_BG = (20, 25, 40)
        self.COLOR_GRID_LINE = (40, 45, 60)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_TEXT_SHADOW = (10, 10, 10)

        # Base palette for tiles and slots
        self.PALETTE = [
            (255, 87, 34), (255, 193, 7), (139, 195, 74), (0, 188, 212),
            (3, 169, 244), (63, 81, 181), (156, 39, 176), (233, 30, 99),
            (0, 255, 128), (255, 128, 0), (128, 0, 255), (0, 128, 255)
        ]
        
        # State variables are initialized in reset()
        self.grid_tiles = []
        self.crystal_pos = [0, 0]
        self.crystal_color = (0,0,0)
        self.target_colors = []
        self.slot_filled = []
        self.particles = []
        self.moves_remaining = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        
        # Initialize state variables
        self.reset()
        
        # Run validation
        self.validate_implementation()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset game state
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.moves_remaining = 3
        self.particles = []

        # Generate level
        self.target_colors = self.np_random.choice(self.PALETTE, size=10, replace=False).tolist()
        self.slot_filled = [False] * 10
        
        self.grid_tiles = [[self.COLOR_GRID_LINE for _ in range(self.GRID_W)] for _ in range(self.GRID_H)]
        
        # Ensure all target colors are on the grid
        available_coords = [(x, y) for x in range(self.GRID_W) for y in range(self.GRID_H)]
        self.np_random.shuffle(available_coords)
        for i, color in enumerate(self.target_colors):
            x, y = available_coords.pop()
            self.grid_tiles[y][x] = tuple(color)

        # Fill rest of grid with random palette colors
        while available_coords:
            x, y = available_coords.pop()
            self.grid_tiles[y][x] = tuple(self.np_random.choice(self.PALETTE, size=1)[0])

        # Set crystal state
        self.crystal_pos = [self.np_random.integers(0, self.GRID_W), self.np_random.integers(0, self.GRID_H)]
        self.crystal_color = self._get_next_target_color()

        # Check for initial match (rare but possible)
        self._check_match()

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        # Unpack factorized action
        movement = action[0]  # 0-4: none/up/down/left/right
        
        self.steps += 1
        self.moves_remaining -= 1

        # Update player position
        dx, dy = 0, 0
        if movement == 1: dy = -1  # Up
        elif movement == 2: dy = 1   # Down
        elif movement == 3: dx = -1  # Left
        elif movement == 4: dx = 1   # Right
        
        new_x = self.crystal_pos[0] + dx
        new_y = self.crystal_pos[1] + dy

        if 0 <= new_x < self.GRID_W and 0 <= new_y < self.GRID_H:
            self.crystal_pos = [new_x, new_y]

        # Check for color match and get immediate reward
        match_reward = self._check_match()
        reward += match_reward
        self.score += match_reward

        # Check for termination
        all_slots_filled = all(self.slot_filled)
        no_moves_left = self.moves_remaining <= 0
        terminated = all_slots_filled or no_moves_left
        self.game_over = terminated

        if terminated and all_slots_filled:
            win_reward = 100
            reward += win_reward
            self.score += win_reward
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

    def _check_match(self):
        """Checks for a color match at the crystal's current position."""
        if self.crystal_color is None:
            return 0
            
        tile_color = self.grid_tiles[self.crystal_pos[1]][self.crystal_pos[0]]
        if tile_color == self.crystal_color:
            try:
                idx = self.target_colors.index(list(tile_color))
                if not self.slot_filled[idx]:
                    self.slot_filled[idx] = True
                    # SFX: Match success chime
                    self._create_particles(self._get_slot_pos(idx), tile_color, 30)
                    self.crystal_color = self._get_next_target_color()
                    return 1 # Reward for one match
            except ValueError:
                pass # Tile color is not a target color
        return 0

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
            "moves_remaining": self.moves_remaining,
            "slots_filled": sum(self.slot_filled)
        }
    
    def _iso_to_cart(self, ix, iy):
        """Converts isometric grid coordinates to cartesian screen coordinates."""
        screen_x = (ix - iy) * self.TILE_W_HALF + self.WIDTH / 2
        screen_y = (ix + iy) * self.TILE_H_HALF + self.HEIGHT / 2 - 80
        return int(screen_x), int(screen_y)

    def _render_text(self, text, pos, font, color, shadow_color=None):
        if shadow_color:
            text_surf_shadow = font.render(text, True, shadow_color)
            self.screen.blit(text_surf_shadow, (pos[0] + 2, pos[1] + 2))
        text_surf = font.render(text, True, color)
        self.screen.blit(text_surf, pos)

    def _render_game(self):
        # Render grid tiles
        for y in range(self.GRID_H):
            for x in range(self.GRID_W):
                self._draw_iso_tile(x, y, self.grid_tiles[y][x])
        
        # Render crystal
        self._draw_iso_cube(self.crystal_pos[0], self.crystal_pos[1], self.crystal_color)
        
        # Update and draw particles
        self._update_and_draw_particles()

    def _render_ui(self):
        # UI Background
        ui_panel_rect = pygame.Rect(0, 0, self.WIDTH, 65)
        pygame.draw.rect(self.screen, (15, 20, 35), ui_panel_rect)
        pygame.draw.line(self.screen, (50, 55, 75), (0, 65), (self.WIDTH, 65), 2)
        
        # Render score and moves
        self._render_text(f"SCORE: {self.score:04d}", (15, 15), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)
        self._render_text(f"MOVES: {self.moves_remaining}", (15, 40), self.font_main, self.COLOR_TEXT, self.COLOR_TEXT_SHADOW)

        # Render target color slots
        slot_size = 20
        slot_padding = 8
        start_x = self.WIDTH - (slot_size + slot_padding) * 10 - 15
        
        self._render_text("TARGETS", (start_x, 10), self.font_small, self.COLOR_TEXT)
        for i, color in enumerate(self.target_colors):
            x = start_x + i * (slot_size + slot_padding)
            y = 30
            rect = pygame.Rect(x, y, slot_size, slot_size)
            
            if self.slot_filled[i]:
                # Draw filled slot (bright with a checkmark)
                pygame.draw.rect(self.screen, color, rect, border_radius=4)
                pygame.draw.rect(self.screen, (255, 255, 255), rect, 2, border_radius=4)
                pygame.draw.line(self.screen, (20,20,20), (x + 5, y + 10), (x + 9, y + 14), 3)
                pygame.draw.line(self.screen, (20,20,20), (x + 9, y + 14), (x + 15, y + 6), 3)
            else:
                # Draw empty slot (darker)
                dark_color = tuple(c // 2 for c in color)
                pygame.draw.rect(self.screen, dark_color, rect, border_radius=4)
                pygame.draw.rect(self.screen, color, rect, 2, border_radius=4)

        # Render "Next Color" indicator
        if self.crystal_color:
            self._render_text("NEXT", (self.WIDTH // 2 - 50, 10), self.font_small, self.COLOR_TEXT)
            pygame.draw.rect(self.screen, self.crystal_color, (self.WIDTH // 2 - 55, 30, 20, 20), border_radius=4)
            pygame.draw.rect(self.screen, (255,255,255), (self.WIDTH // 2 - 55, 30, 20, 20), 2, border_radius=4)

    def _draw_iso_tile(self, ix, iy, color):
        p1 = self._iso_to_cart(ix, iy)
        p2 = self._iso_to_cart(ix + 1, iy)
        p3 = self._iso_to_cart(ix + 1, iy + 1)
        p4 = self._iso_to_cart(ix, iy + 1)
        points = [p1, p2, p3, p4]
        
        pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GRID_LINE)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)

    def _draw_iso_cube(self, ix, iy, color):
        if color is None: return
        
        cx, cy = self._iso_to_cart(ix + 0.5, iy + 0.5)
        cy -= self.TILE_H_HALF # Elevate cube
        
        # Colors for faces
        top_color = color
        left_color = tuple(max(0, c - 40) for c in color)
        right_color = tuple(max(0, c - 80) for c in color)
        
        # Points
        p_top = (cx, cy - self.TILE_H_HALF)
        p_mid = (cx, cy)
        p_left = (cx - self.TILE_W_HALF, cy - self.TILE_H_HALF)
        p_right = (cx + self.TILE_W_HALF, cy - self.TILE_H_HALF)
        p_bot = (cx, cy + self.TILE_H_HALF)
        p_bot_left = (cx - self.TILE_W_HALF, cy)
        p_bot_right = (cx + self.TILE_W_HALF, cy)
        
        # Draw faces (back to front)
        # Right face
        pygame.gfxdraw.aapolygon(self.screen, [p_mid, p_right, p_bot_right, p_bot], right_color)
        pygame.gfxdraw.filled_polygon(self.screen, [p_mid, p_right, p_bot_right, p_bot], right_color)
        # Left face
        pygame.gfxdraw.aapolygon(self.screen, [p_mid, p_left, p_bot_left, p_bot], left_color)
        pygame.gfxdraw.filled_polygon(self.screen, [p_mid, p_left, p_bot_left, p_bot], left_color)
        # Top face
        pygame.gfxdraw.aapolygon(self.screen, [p_top, p_right, p_mid, p_left], top_color)
        pygame.gfxdraw.filled_polygon(self.screen, [p_top, p_right, p_mid, p_left], top_color)

    def _get_next_target_color(self):
        for i, filled in enumerate(self.slot_filled):
            if not filled:
                return tuple(self.target_colors[i])
        return None # All slots filled

    def _get_slot_pos(self, index):
        slot_size = 20
        slot_padding = 8
        start_x = self.WIDTH - (slot_size + slot_padding) * 10 - 15
        x = start_x + index * (slot_size + slot_padding) + slot_size // 2
        y = 30 + slot_size // 2
        return x, y

    def _create_particles(self, pos, color, count):
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifespan = self.np_random.integers(15, 30)
            self.particles.append({'pos': list(pos), 'vel': vel, 'lifespan': lifespan, 'color': color})

    def _update_and_draw_particles(self):
        active_particles = []
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['lifespan'] -= 1
            if p['lifespan'] > 0:
                radius = int(p['lifespan'] / 5)
                if radius > 0:
                    pygame.gfxdraw.filled_circle(self.screen, int(p['pos'][0]), int(p['pos'][1]), radius, p['color'])
                active_particles.append(p)
        self.particles = active_particles

    def validate_implementation(self):
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(info, dict)
        assert info['moves_remaining'] == 3
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.HEIGHT, self.WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert not trunc
        assert isinstance(info, dict)
        assert info['moves_remaining'] == 2
        
        # Test termination after 3 steps
        self.reset()
        for _ in range(3):
            obs, reward, term, trunc, info = self.step(self.action_space.sample())
        assert term is True
        
        print("✓ Implementation validated successfully")

    def close(self):
        pygame.quit()

if __name__ == '__main__':
    env = GameEnv()
    obs, info = env.reset()
    terminated = False
    
    # Mapping from Pygame keys to environment actions
    key_to_action = {
        pygame.K_UP: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3,
        pygame.K_RIGHT: 4,
    }
    
    # Create a display window
    display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
    pygame.display.set_caption("Crystal Cavern")
    
    running = True
    while running:
        action = np.array([0, 0, 0])  # Default to no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action[0] = key_to_action[event.key]
                    obs, reward, terminated, truncated, info = env.step(action)
                    print(f"Action: {action}, Reward: {reward}, Score: {info['score']}, Terminated: {terminated}")
                
                if event.key == pygame.K_r:
                    print("--- RESETTING ENVIRONMENT ---")
                    obs, info = env.reset()
                    terminated = False

        # Render the environment to the display
        frame = env._get_observation()
        frame = np.transpose(frame, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        display_screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        if terminated:
            print("--- GAME OVER ---")
            print(f"Final Score: {info['score']}")
            # Wait for 'R' to reset
            waiting_for_reset = True
            while waiting_for_reset:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting_for_reset = False
                        running = False
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        print("--- RESETTING ENVIRONMENT ---")
                        obs, info = env.reset()
                        terminated = False
                        waiting_for_reset = False
    
    env.close()