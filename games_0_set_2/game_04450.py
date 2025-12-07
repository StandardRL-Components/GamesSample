
# Generated: 2025-08-28T02:26:38.535454
# Source Brief: brief_04450.md
# Brief Index: 4450

        
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
        "Controls: Use ↑↓←→ to move your character on the grid. "
        "Collect 15 crystals in 20 moves."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic isometric puzzle game. Plan your path to collect all "
        "the crystals before you run out of moves."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 10
    GRID_HEIGHT = 8
    TILE_WIDTH_HALF = 28
    TILE_HEIGHT_HALF = 14
    MAX_MOVES = 20
    CRYSTAL_TARGET = 15
    MAX_STEPS = 1000

    # Colors
    COLOR_BG = (44, 62, 80)  # Dark Blue/Gray
    COLOR_GRID = (52, 73, 94)
    COLOR_PLAYER = (241, 196, 15)  # Yellow
    COLOR_PLAYER_SHADOW = (30, 40, 50)
    CRYSTAL_COLORS = [(26, 188, 156), (155, 89, 182)] # Teal, Purple
    COLOR_TEXT = (236, 240, 241)
    COLOR_UI_BG = (40, 55, 71, 180)

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # EXACT spaces:
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # Pygame setup
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_main = pygame.font.SysFont("monospace", 20, bold=True)
        self.font_large = pygame.font.SysFont("monospace", 48, bold=True)

        # Centering offset for the grid
        self.grid_offset_x = self.SCREEN_WIDTH / 2
        self.grid_offset_y = self.SCREEN_HEIGHT / 2 - (self.GRID_HEIGHT * self.TILE_HEIGHT_HALF) / 2 + 30
        
        # State variables are initialized in reset()
        self.player_pos = [0, 0]
        self.crystals = []
        self.particles = []
        self.moves_left = 0
        self.crystals_collected = 0
        self.score = 0
        self.steps = 0
        self.game_over = False
        self.win = False
        self.last_reward = 0

        # Initialize state variables
        self.reset()

        self.validate_implementation()
        
    def _grid_to_screen(self, x, y):
        """Converts grid coordinates to isometric screen coordinates."""
        screen_x = self.grid_offset_x + (x - y) * self.TILE_WIDTH_HALF
        screen_y = self.grid_offset_y + (x + y) * self.TILE_HEIGHT_HALF
        return int(screen_x), int(screen_y)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.game_over = False
        self.win = False
        self.moves_left = self.MAX_MOVES
        self.crystals_collected = 0
        self.particles = []
        self.last_reward = 0

        # Place player in the center
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]

        # Generate crystal positions
        self.crystals = []
        possible_positions = []
        for x in range(self.GRID_WIDTH):
            for y in range(self.GRID_HEIGHT):
                if [x, y] != self.player_pos:
                    possible_positions.append([x, y])
        
        indices = self.np_random.choice(
            len(possible_positions), self.CRYSTAL_TARGET, replace=False
        )
        for i in indices:
            self.crystals.append(possible_positions[i])

        return self._get_observation(), self._get_info()

    def step(self, action):
        movement = action[0]  # 0-4: none/up/down/left/right
        
        reward = 0
        terminated = False
        self.steps += 1

        moved = False
        if not self.game_over and movement != 0:
            moved = True
            self.moves_left -= 1
            
            # Update player position based on action
            dx, dy = 0, 0
            if movement == 1:  # Up
                dy = -1
            elif movement == 2:  # Down
                dy = 1
            elif movement == 3:  # Left
                dx = -1
            elif movement == 4:  # Right
                dx = 1

            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

            # Boundary checks
            if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
                self.player_pos = new_pos

        # Check for crystal collection
        if moved:
            for crystal_pos in self.crystals:
                if self.player_pos == crystal_pos:
                    # SFX: Crystal collect sound
                    self.crystals.remove(crystal_pos)
                    self.crystals_collected += 1
                    self.score += 1
                    reward += 1
                    
                    # Create particle explosion
                    screen_pos = self._grid_to_screen(crystal_pos[0], crystal_pos[1])
                    self._create_explosion(screen_pos, 30, random.choice(self.CRYSTAL_COLORS))
                    
                    # Check for 15th crystal bonus
                    if self.crystals_collected == self.CRYSTAL_TARGET:
                        reward += 10
                        self.score += 10
                    break
        
        # Check termination conditions
        if self.crystals_collected >= self.CRYSTAL_TARGET:
            # SFX: Win jingle
            terminated = True
            self.game_over = True
            self.win = True
            reward += 100
            self.score += 100
        elif self.moves_left <= 0:
            # SFX: Lose sound
            terminated = True
            self.game_over = True
            self.win = False
            reward -= 100
            self.score -= 100
        
        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        self.last_reward = reward
        return (
            self._get_observation(),
            reward,
            terminated,
            False,  # truncated always False
            self._get_info()
        )

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "moves_left": self.moves_left,
            "crystals_collected": self.crystals_collected,
        }

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        
        self._render_game()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_game(self):
        # Draw grid
        for y in range(self.GRID_HEIGHT + 1):
            start = self._grid_to_screen(0, y)
            end = self._grid_to_screen(self.GRID_WIDTH, y)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)
        for x in range(self.GRID_WIDTH + 1):
            start = self._grid_to_screen(x, 0)
            end = self._grid_to_screen(x, self.GRID_HEIGHT)
            pygame.draw.line(self.screen, self.COLOR_GRID, start, end, 1)

        # Draw crystals
        for i, pos in enumerate(self.crystals):
            self._draw_crystal(pos[0], pos[1], i)
            
        # Draw player
        self._draw_player()

    def _draw_iso_poly(self, surface, color, points, offset_y=0):
        """Helper to draw an antialiased polygon with a vertical offset."""
        offset_points = [(p[0], p[1] + offset_y) for p in points]
        pygame.gfxdraw.aapolygon(surface, offset_points, color)
        pygame.gfxdraw.filled_polygon(surface, offset_points, color)

    def _draw_player(self):
        px, py = self._grid_to_screen(self.player_pos[0], self.player_pos[1])
        
        # Bobbing animation
        bob = math.sin(self.steps * 0.15) * 4 - 4
        
        # Shadow
        shadow_radius = self.TILE_WIDTH_HALF * 0.5
        shadow_surface = pygame.Surface((shadow_radius * 2, shadow_radius * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, (0, 0, 0, 60), shadow_surface.get_rect())
        self.screen.blit(shadow_surface, (px - shadow_radius, py - shadow_radius/2))
        
        # Player body
        radius = 10
        player_y = py + bob - radius
        pygame.gfxdraw.filled_circle(self.screen, px, int(player_y), radius, self.COLOR_PLAYER)
        pygame.gfxdraw.aacircle(self.screen, px, int(player_y), radius, self.COLOR_PLAYER)

    def _draw_crystal(self, x, y, index):
        cx, cy = self._grid_to_screen(x, y)
        
        # Bobbing and spinning animation
        bob = math.sin(self.steps * 0.1 + index) * 3
        spin = self.steps * 0.05 + index
        
        size = self.TILE_WIDTH_HALF * 0.4
        
        # Shadow
        shadow_radius = size * 0.8
        shadow_surface = pygame.Surface((shadow_radius * 2, shadow_radius * 2), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surface, (0, 0, 0, 50), shadow_surface.get_rect())
        self.screen.blit(shadow_surface, (cx - shadow_radius, cy - shadow_radius/2 + 2))
        
        # Crystal shape (diamond)
        color = self.CRYSTAL_COLORS[index % len(self.CRYSTAL_COLORS)]
        points = [
            (cx, cy + bob - size),
            (cx + size * math.cos(spin), cy + bob),
            (cx, cy + bob + size),
            (cx - size * math.cos(spin), cy + bob),
        ]
        
        pygame.gfxdraw.aapolygon(self.screen, points, color)
        pygame.gfxdraw.filled_polygon(self.screen, points, color)
        
        # Highlight
        highlight_color = (255, 255, 255, 150)
        highlight_points = [
            (cx, cy + bob - size),
            (cx + size * math.cos(spin) * 0.5, cy + bob),
            (cx, cy + bob),
        ]
        pygame.gfxdraw.filled_polygon(self.screen, highlight_points, highlight_color)
        
    def _render_particles(self):
        # Update and draw particles
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1  # Gravity
            p['life'] -= 1
            if p['life'] <= 0:
                self.particles.remove(p)
            else:
                alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
                color = p['color'] + (alpha,)
                size = max(1, int(p['size'] * (p['life'] / p['max_life'])))
                pygame.draw.circle(self.screen, color, p['pos'], size)

    def _create_explosion(self, pos, num_particles, color):
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                'pos': list(pos),
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed - 2],
                'life': random.randint(20, 40),
                'max_life': 40,
                'color': color,
                'size': random.randint(2, 5)
            })

    def _render_ui(self):
        # UI Background panels
        s = pygame.Surface((180, 50), pygame.SRCALPHA)
        s.fill(self.COLOR_UI_BG)
        self.screen.blit(s, (10, 10))
        self.screen.blit(s, (self.SCREEN_WIDTH - 190, 10))

        # Crystals collected
        crystal_text = self.font_main.render(f"CRYSTALS", True, self.COLOR_TEXT)
        self.screen.blit(crystal_text, (20, 15))
        crystal_count_text = self.font_main.render(f"{self.crystals_collected} / {self.CRYSTAL_TARGET}", True, self.COLOR_TEXT)
        self.screen.blit(crystal_count_text, (20, 35))

        # Moves left
        moves_text = self.font_main.render(f"MOVES LEFT", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (self.SCREEN_WIDTH - 180, 15))
        moves_count_text = self.font_main.render(f"{self.moves_left}", True, self.COLOR_TEXT)
        self.screen.blit(moves_count_text, (self.SCREEN_WIDTH - 180, 35))

        # Game Over message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "YOU WIN!" if self.win else "GAME OVER"
            color = (46, 204, 113) if self.win else (231, 76, 60)
            
            end_text = self.font_large.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
        # Test action space
        assert self.action_space.shape == (3,)
        assert self.action_space.nvec.tolist() == [5, 2, 2]
        
        # Test observation space  
        test_obs = self._get_observation()
        assert test_obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert test_obs.dtype == np.uint8
        
        # Test reset
        obs, info = self.reset()
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(info, dict)
        
        # Test step
        test_action = self.action_space.sample()
        obs, reward, term, trunc, info = self.step(test_action)
        assert obs.shape == (self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3)
        assert isinstance(reward, (int, float))
        assert isinstance(term, bool)
        assert trunc == False
        assert isinstance(info, dict)
        
        print("✓ Implementation validated successfully")

if __name__ == '__main__':
    # This block allows you to play the game with keyboard controls
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Pygame setup for human play
    pygame.display.set_caption("Isometric Crystal Collector")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    while running:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r: # Press 'R' to reset
                    obs, info = env.reset()
                    done = False
                if not done:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
        
        if action[0] != 0 or env.auto_advance:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")

        # Render the observation to the display
        frame = np.transpose(obs, (1, 0, 2))
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit to 30 FPS for human play

    pygame.quit()