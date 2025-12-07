
# Generated: 2025-08-27T13:57:48.728761
# Source Brief: brief_00542.md
# Brief Index: 542

        
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
        "Controls: ↑↓←→ to move. Collect all 10 gems before your 20 moves run out."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Collect 10 gems within 20 moves by navigating the grid."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    # --- Constants ---
    # Colors
    COLOR_BG = (15, 23, 42)
    COLOR_GRID = (51, 65, 85)
    COLOR_PLAYER = (253, 224, 71)
    COLOR_TEXT = (241, 245, 249)
    COLOR_TEXT_SHADOW = (15, 23, 42)
    GEM_COLORS = [
        (239, 68, 68),   # Red
        (59, 130, 246),  # Blue
        (34, 197, 94),   # Green
        (168, 85, 247),  # Purple
        (249, 115, 22),  # Orange
    ]

    # Game settings
    SCREEN_WIDTH = 640
    SCREEN_HEIGHT = 400
    GRID_WIDTH = 16
    GRID_HEIGHT = 10
    CELL_SIZE = 40
    MAX_MOVES = 20
    GEM_TARGET = 10
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        
        # Fonts
        self.font_ui = pygame.font.SysFont("monospace", 18, bold=True)
        self.font_msg = pygame.font.SysFont("monospace", 48, bold=True)
        
        # Game state variables (initialized in reset)
        self.player_pos = None
        self.gems = None
        self.particles = None
        self.moves_left = None
        self.gems_collected = None
        self.score = None
        self.game_over = None
        self.steps = None
        self.animation_timer = 0

        self.reset()
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        # Generate gem locations
        self.gems = []
        possible_locations = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        possible_locations.remove(tuple(self.player_pos))
        
        gem_indices = self.np_random.choice(len(possible_locations), self.GEM_TARGET, replace=False)
        gem_locations = [possible_locations[i] for i in gem_indices]
        
        for i, pos in enumerate(gem_locations):
            self.gems.append({
                "pos": list(pos),
                "color": self.np_random.choice(self.GEM_COLORS),
                "spawn_time": i * 5 # Stagger animation start
            })
            
        self.particles = []
        self.moves_left = self.MAX_MOVES
        self.gems_collected = 0
        self.score = 0
        self.game_over = False
        self.steps = 0
        self.animation_timer = 0
        
        return self._get_observation(), self._get_info()
    
    def step(self, action):
        reward = 0
        self.animation_timer += 1
        
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        
        # Process movement
        if movement != 0: # 0 is no-op
            self.steps += 1
            self.moves_left -= 1
            
            dx, dy = 0, 0
            if movement == 1: dy = -1  # Up
            elif movement == 2: dy = 1   # Down
            elif movement == 3: dx = -1  # Left
            elif movement == 4: dx = 1   # Right
            
            new_pos = [self.player_pos[0] + dx, self.player_pos[1] + dy]
            
            # Clamp to grid boundaries
            new_pos[0] = max(0, min(self.GRID_WIDTH - 1, new_pos[0]))
            new_pos[1] = max(0, min(self.GRID_HEIGHT - 1, new_pos[1]))
            
            self.player_pos = new_pos

        # Check for gem collection
        gem_to_remove = None
        for gem in self.gems:
            if self.player_pos == gem["pos"]:
                gem_to_remove = gem
                break
        
        if gem_to_remove:
            self.gems.remove(gem_to_remove)
            self.gems_collected += 1
            
            # sfx: gem_collect.wav
            self._create_particles(self._grid_to_pixel(gem_to_remove["pos"]), gem_to_remove["color"])

            if self.gems_collected == self.GEM_TARGET:
                reward += 10 # Bonus for the last gem
            else:
                reward += 1

        self.score += reward

        # Update particles
        self._update_particles()
        
        # Check for termination
        win_condition = self.gems_collected >= self.GEM_TARGET
        loss_condition = self.moves_left <= 0
        terminated = win_condition or loss_condition
        
        if terminated:
            self.game_over = True
            if win_condition:
                reward += 100 # Win bonus
                # sfx: win_jingle.wav
            elif loss_condition:
                reward -= 100 # Loss penalty
                # sfx: lose_sound.wav
        
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
            "moves_left": self.moves_left,
            "gems_collected": self.gems_collected,
        }
        
    def _render_game(self):
        self._draw_grid()
        self._draw_gems()
        self._draw_particles()
        self._draw_player()

    def _render_ui(self):
        # Moves Left Display
        moves_text = self.font_ui.render(f"Moves: {self.moves_left}", True, self.COLOR_TEXT)
        self._draw_text_shadow(moves_text, (10, 10))

        # Gems Collected Display
        gems_text = self.font_ui.render(f"Gems: {self.gems_collected}/{self.GEM_TARGET}", True, self.COLOR_TEXT)
        gems_text_rect = gems_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self._draw_text_shadow(gems_text, gems_text_rect.topleft)

        # Game Over Message
        if self.game_over:
            if self.gems_collected >= self.GEM_TARGET:
                msg = "YOU WIN!"
                color = (134, 239, 172) # Bright Green
            else:
                msg = "GAME OVER"
                color = (252, 165, 165) # Bright Red
                
            msg_surf = self.font_msg.render(msg, True, color)
            msg_rect = msg_surf.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self._draw_text_shadow(msg_surf, msg_rect.topleft)

    def _draw_text_shadow(self, surface, pos):
        x, y = pos
        shadow_surf = surface.copy()
        shadow_surf.set_colorkey(surface.get_colorkey())
        shadow_alpha = pygame.mask.from_surface(surface).to_surface(setcolor=self.COLOR_TEXT_SHADOW, unsetcolor=(0,0,0,0))
        self.screen.blit(shadow_alpha, (x + 2, y + 2))
        self.screen.blit(surface, pos)

    def _grid_to_pixel(self, grid_pos):
        x = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        y = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(x), int(y)

    def _draw_grid(self):
        for x in range(0, self.SCREEN_WIDTH, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (x, 0), (x, self.SCREEN_HEIGHT))
        for y in range(0, self.SCREEN_HEIGHT, self.CELL_SIZE):
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, y), (self.SCREEN_WIDTH, y))

    def _draw_player(self):
        px, py = self._grid_to_pixel(self.player_pos)
        player_rect = pygame.Rect(px - self.CELL_SIZE // 3, py - self.CELL_SIZE // 3, self.CELL_SIZE * 2/3, self.CELL_SIZE * 2/3)
        pygame.draw.rect(self.screen, self.COLOR_PLAYER, player_rect, border_radius=4)
        pygame.draw.rect(self.screen, (254, 249, 195), player_rect.inflate(-4, -4), border_radius=3)

    def _draw_gems(self):
        for gem in self.gems:
            px, py = self._grid_to_pixel(gem["pos"])
            
            # Pulsing animation
            pulse = math.sin((self.animation_timer + gem["spawn_time"]) * 0.1) * 2
            radius = int(self.CELL_SIZE / 3 + pulse)
            
            # Use gfxdraw for anti-aliasing
            pygame.gfxdraw.aacircle(self.screen, px, py, radius, gem["color"])
            pygame.gfxdraw.filled_circle(self.screen, px, py, radius, gem["color"])
            
            # Inner shine
            shine_radius = max(0, radius // 3)
            pygame.gfxdraw.aacircle(self.screen, px - radius // 4, py - radius // 4, shine_radius, (255, 255, 255, 100))
            pygame.gfxdraw.filled_circle(self.screen, px - radius // 4, py - radius // 4, shine_radius, (255, 255, 255, 100))

    def _create_particles(self, pos, color):
        for _ in range(20):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            self.particles.append({
                "pos": list(pos),
                "vel": vel,
                "radius": self.np_random.uniform(3, 7),
                "color": color,
                "life": self.np_random.integers(20, 40)
            })

    def _update_particles(self):
        for p in self.particles[:]:
            p["pos"][0] += p["vel"][0]
            p["pos"][1] += p["vel"][1]
            p["vel"][0] *= 0.98 # friction
            p["vel"][1] *= 0.98 # friction
            p["life"] -= 1
            p["radius"] -= 0.1
            if p["life"] <= 0 or p["radius"] <= 0:
                self.particles.remove(p)

    def _draw_particles(self):
        for p in self.particles:
            pos = (int(p["pos"][0]), int(p["pos"][1]))
            radius = int(max(0, p["radius"]))
            
            # Create a temporary surface for transparency
            alpha = max(0, min(255, int(p["life"] * 6)))
            color_with_alpha = p["color"] + (alpha,)
            
            # Use simple circles for performance
            pygame.draw.circle(self.screen, p["color"], pos, radius)

    def close(self):
        pygame.quit()

    def validate_implementation(self):
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

# Example usage to run and display the game
if __name__ == "__main__":
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()

    # Setup Pygame window for human play
    pygame.display.set_caption(env.game_description)
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    running = True
    terminated = False

    print(env.user_guide)

    while running:
        action = [0, 0, 0] # Default action: no-op

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_r: # Reset on 'r'
                    obs, info = env.reset()
                    terminated = False
                
                # Only register one key press per frame for this turn-based game
                if not terminated:
                    if event.key == pygame.K_UP:
                        action[0] = 1
                    elif event.key == pygame.K_DOWN:
                        action[0] = 2
                    elif event.key == pygame.K_LEFT:
                        action[0] = 3
                    elif event.key == pygame.K_RIGHT:
                        action[0] = 4
                    
                    # If any move key was pressed, step the environment
                    if action[0] != 0:
                        obs, reward, terminated, truncated, info = env.step(action)
                        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Info: {info}")

        # Draw the observation from the environment to the screen
        frame_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(frame_surface, (0, 0))
        
        pygame.display.flip()
        clock.tick(30) # Limit FPS

    env.close()