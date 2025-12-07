
# Generated: 2025-08-28T02:21:56.280281
# Source Brief: brief_04422.md
# Brief Index: 4422

        
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
        "Controls: Use arrow keys (↑, ↓, ←, →) to move your gem one square at a time."
    )

    game_description = (
        "A strategic puzzle game. Collect all the gems on the grid before you run out of moves. Plan your path carefully to clear each stage and maximize your score."
    )

    auto_advance = False

    # --- Constants ---
    SCREEN_WIDTH, SCREEN_HEIGHT = 640, 400
    GRID_WIDTH, GRID_HEIGHT = 10, 10
    MAX_EPISODE_STEPS = 1000
    MAX_STAGES = 3

    # --- Colors ---
    COLOR_BG = (26, 35, 126)          # Dark Indigo
    COLOR_GRID = (40, 53, 147)        # Darker Blue
    COLOR_PLAYER = (244, 67, 54)      # Bright Red
    COLOR_PLAYER_GLOW = (244, 67, 54, 100)
    COLOR_GEMS = [(76, 175, 80), (255, 235, 59), (156, 39, 176)] # Green, Yellow, Purple
    COLOR_TEXT = (255, 255, 255)
    COLOR_UI_BG = (0, 0, 0, 128)

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
        
        try:
            self.font_large = pygame.font.Font(None, 48)
            self.font_medium = pygame.font.Font(None, 32)
            self.font_small = pygame.font.Font(None, 24)
        except IOError:
            self.font_large = pygame.font.SysFont("sans-serif", 48)
            self.font_medium = pygame.font.SysFont("sans-serif", 32)
            self.font_small = pygame.font.SysFont("sans-serif", 24)
        
        # Game state variables are initialized in reset()
        self.player_pos = None
        self.gems = None
        self.moves_remaining = None
        self.stage = None
        self.initial_gems_this_stage = None
        self.score = None
        self.steps = None
        self.game_over = None
        self.game_won = None
        self.particles = []
        self.flash_effects = []
        
        self.reset()
        
        self.validate_implementation()

    def _setup_stage(self):
        """Initializes the state for the current stage."""
        num_gems = 20 + (self.stage - 1) * 2
        self.initial_gems_this_stage = num_gems
        self.moves_remaining = 50
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        possible_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        possible_cells.remove(tuple(self.player_pos))
        
        gem_indices = self.np_random.choice(len(possible_cells), size=num_gems, replace=False)
        self.gems = []
        for i in gem_indices:
            pos = possible_cells[i]
            color = self.np_random.choice(self.COLOR_GEMS)
            self.gems.append({"pos": list(pos), "color": color})

        self.particles.clear()
        self.flash_effects.clear()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.game_won = False
        self.stage = 1
        
        self._setup_stage()
        
        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0
        terminated = False
        
        moved = False
        if movement in [1, 2, 3, 4]: # Up, Down, Left, Right
            dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[movement]
            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]

            if 0 <= new_pos[0] < self.GRID_WIDTH and 0 <= new_pos[1] < self.GRID_HEIGHT:
                self.player_pos = new_pos
                moved = True
        
        # A no-op (action 0) or a valid move both consume a turn
        if movement == 0 or moved:
            self.moves_remaining -= 1
            # Sound: Move sfx
            
            gem_collected = False
            for i, gem in enumerate(self.gems):
                if self.player_pos == gem["pos"]:
                    # Risky play bonus check (before removing the gem)
                    adjacent_gems = 0
                    for other_gem in self.gems:
                        if gem == other_gem: continue
                        dist = abs(gem["pos"][0] - other_gem["pos"][0]) + abs(gem["pos"][1] - other_gem["pos"][1])
                        if dist == 1:
                            adjacent_gems += 1
                    if adjacent_gems >= 2:
                        reward += 5
                        # Sound: Risky collect bonus sfx

                    self.score += 1
                    reward += 1
                    self._create_particles(gem["pos"], gem["color"])
                    self.flash_effects.append({"pos": gem["pos"], "timer": 5})
                    self.gems.pop(i)
                    gem_collected = True
                    # Sound: Gem collect sfx
                    break

            if not gem_collected and (movement == 0 or moved):
                reward += -0.2 * self.initial_gems_this_stage

        self._update_particles()
        
        # Check for stage clear
        if not self.gems:
            self.score += 100
            reward += 100
            self.stage += 1
            if self.stage > self.MAX_STAGES:
                self.game_over = True
                self.game_won = True
                terminated = True
                # Sound: Game win fanfare
            else:
                self._setup_stage()
                # Sound: Stage clear sfx

        # Check for loss condition
        if self.moves_remaining <= 0 and not self.game_over:
            self.score -= 100
            reward -= 100
            self.game_over = True
            terminated = True
            # Sound: Game over sfx

        self.steps += 1
        if self.steps >= self.MAX_EPISODE_STEPS:
            terminated = True
        
        return (
            self._get_observation(),
            reward,
            terminated,
            False,
            self._get_info()
        )

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
            "stage": self.stage,
            "moves_remaining": self.moves_remaining,
            "gems_remaining": len(self.gems)
        }

    def _render_game(self):
        # Calculate grid geometry
        grid_pixel_width = self.SCREEN_HEIGHT - 40
        self.cell_size = grid_pixel_width // self.GRID_HEIGHT
        offset_x = (self.SCREEN_WIDTH - self.cell_size * self.GRID_WIDTH) / 2
        offset_y = (self.SCREEN_HEIGHT - self.cell_size * self.GRID_HEIGHT) / 2
        
        # Draw grid lines
        for i in range(self.GRID_WIDTH + 1):
            x = int(offset_x + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, offset_y), (x, offset_y + self.GRID_HEIGHT * self.cell_size), 2)
        for i in range(self.GRID_HEIGHT + 1):
            y = int(offset_y + i * self.cell_size)
            pygame.draw.line(self.screen, self.COLOR_GRID, (offset_x, y), (offset_x + self.GRID_WIDTH * self.cell_size, y), 2)

        # Draw particles
        self._draw_particles(offset_x, offset_y)

        # Draw gems
        gem_radius = int(self.cell_size * 0.35)
        for gem in self.gems:
            center_x = int(offset_x + gem["pos"][0] * self.cell_size + self.cell_size / 2)
            center_y = int(offset_y + gem["pos"][1] * self.cell_size + self.cell_size / 2)
            self._draw_polygon_gem(self.screen, gem["color"], (center_x, center_y), gem_radius, 6)
        
        # Draw flash effects
        for effect in self.flash_effects:
            if effect["timer"] > 0:
                center_x = int(offset_x + effect["pos"][0] * self.cell_size + self.cell_size / 2)
                center_y = int(offset_y + effect["pos"][1] * self.cell_size + self.cell_size / 2)
                flash_radius = int(self.cell_size * 0.5 * (1 - effect["timer"]/5))
                pygame.gfxdraw.filled_circle(self.screen, center_x, center_y, flash_radius, (255, 255, 255, 150))
                effect["timer"] -= 1
        self.flash_effects = [e for e in self.flash_effects if e["timer"] > 0]


        # Draw player
        player_radius = int(self.cell_size * 0.4)
        player_center_x = int(offset_x + self.player_pos[0] * self.cell_size + self.cell_size / 2)
        player_center_y = int(offset_y + self.player_pos[1] * self.cell_size + self.cell_size / 2)

        # Glow effect
        glow_surf = pygame.Surface((player_radius * 2.5, player_radius * 2.5), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (player_radius*1.25, player_radius*1.25), player_radius * 1.2)
        self.screen.blit(glow_surf, (player_center_x - player_radius*1.25, player_center_y - player_radius*1.25))

        self._draw_polygon_gem(self.screen, self.COLOR_PLAYER, (player_center_x, player_center_y), player_radius, 8)

    def _draw_polygon_gem(self, surface, color, center, radius, sides):
        """Draws a multi-faceted gem shape."""
        points = []
        for i in range(sides):
            angle = math.radians(360 / sides * i)
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((int(x), int(y)))
        
        pygame.gfxdraw.aapolygon(surface, points, color)
        pygame.gfxdraw.filled_polygon(surface, points, color)

        # Highlight
        highlight_color = tuple(min(255, c + 60) for c in color)
        highlight_points = []
        for i in range(sides):
            angle = math.radians(360 / sides * i)
            x = center[0] + radius * 0.5 * math.cos(angle)
            y = center[1] - radius * 0.2 + radius * 0.5 * math.sin(angle)
            highlight_points.append((int(x), int(y)))
        pygame.gfxdraw.filled_polygon(surface, highlight_points, highlight_color)

    def _render_ui(self):
        # Moves remaining
        moves_text = self.font_medium.render(f"Moves: {self.moves_remaining}", True, self.COLOR_TEXT)
        self.screen.blit(moves_text, (20, 10))

        # Score
        score_text = self.font_medium.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        score_rect = score_text.get_rect(topright=(self.SCREEN_WIDTH - 20, 10))
        self.screen.blit(score_text, score_rect)
        
        # Stage
        stage_text = self.font_small.render(f"Stage: {self.stage}/{self.MAX_STAGES}", True, self.COLOR_TEXT)
        stage_rect = stage_text.get_rect(centerx=self.SCREEN_WIDTH / 2, bottom=self.SCREEN_HEIGHT - 10)
        self.screen.blit(stage_text, stage_rect)

        # Game Over / Win message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill(self.COLOR_UI_BG)
            
            if self.game_won:
                msg = "YOU WIN!"
            else:
                msg = "GAME OVER"
                
            end_text = self.font_large.render(msg, True, self.COLOR_TEXT)
            end_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            
            self.screen.blit(overlay, (0, 0))
            self.screen.blit(end_text, end_rect)

    def _create_particles(self, grid_pos, color):
        cell_size = (self.SCREEN_HEIGHT - 40) // self.GRID_HEIGHT
        offset_x = (self.SCREEN_WIDTH - cell_size * self.GRID_WIDTH) / 2
        offset_y = (self.SCREEN_HEIGHT - cell_size * self.GRID_HEIGHT) / 2
        
        px = offset_x + grid_pos[0] * cell_size + cell_size / 2
        py = offset_y + grid_pos[1] * cell_size + cell_size / 2

        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            vel = [speed * math.cos(angle), speed * math.sin(angle)]
            radius = self.np_random.uniform(2, 5)
            life = self.np_random.integers(15, 30)
            self.particles.append({"pos": [px, py], "vel": vel, "radius": radius, "color": color, "life": life})

    def _update_particles(self):
        for p in self.particles:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["radius"] *= 0.95
            p["life"] -= 1
        self.particles = [p for p in self.particles if p["life"] > 0 and p["radius"] > 0.5]

    def _draw_particles(self, offset_x, offset_y):
        for p in self.particles:
            alpha = int(255 * (p["life"] / 30))
            color = (*p["color"], alpha)
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            radius = int(p["radius"])
            
            temp_surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA)
            pygame.draw.circle(temp_surf, color, (radius, radius), radius)
            self.screen.blit(temp_surf, (pos[0]-radius, pos[1]-radius))

    def validate_implementation(self):
        """
        Call this at the end of __init__ to verify implementation.
        """
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
    # This block allows you to play the game manually
    env = GameEnv()
    obs, info = env.reset()
    done = False
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((GameEnv.SCREEN_WIDTH, GameEnv.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    total_reward = 0
    
    print(GameEnv.game_description)
    print(GameEnv.user_guide)

    while not done:
        # --- Action mapping for human play ---
        action = [0, 0, 0] # Default to no-op
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]: action[0] = 1
        elif keys[pygame.K_DOWN]: action[0] = 2
        elif keys[pygame.K_LEFT]: action[0] = 3
        elif keys[pygame.K_RIGHT]: action[0] = 4
        
        # In auto_advance=False mode, we only step when an event occurs
        move_made = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_UP, pygame.K_DOWN, pygame.K_LEFT, pygame.K_RIGHT]:
                    move_made = True
                if event.key == pygame.K_q:
                    done = True
        
        if move_made:
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            print(f"Action: {action}, Reward: {reward:.2f}, Total Reward: {total_reward:.2f}, Info: {info}")
            if terminated or truncated:
                done = True

        # --- Rendering ---
        # The observation is already a rendered frame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        clock.tick(30) # Limit FPS for human play

    print("Game Over!")
    print(f"Final Score: {info.get('score', 0)}")
    pygame.quit()