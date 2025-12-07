
# Generated: 2025-08-27T13:18:09.183557
# Source Brief: brief_00322.md
# Brief Index: 322

        
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random
from collections import deque
import os
import pygame


class GameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Arrow keys to move on the isometric grid. Avoid red traps and collect all 20 white crystals."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "Navigate the isometric Crystal Caverns, collecting crystals while dodging deadly traps to amass a high score."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False
    
    def __init__(self, render_mode="rgb_array"):
        super().__init__()
        
        # --- Gymnasium Spaces ---
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])
        
        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen_width = 640
        self.screen_height = 400
        self.screen = pygame.Surface((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- Game Constants ---
        self.grid_size = (10, 10)
        self.num_crystals = 20
        self.num_traps = 5
        self.max_steps = 1000
        self.win_crystal_count = 20

        # --- Visual Constants ---
        self.tile_width = 48
        self.tile_height = 24
        self.origin_x = self.screen_width // 2
        self.origin_y = 100

        self.COLOR_BG = (20, 25, 40)
        self.COLOR_TILE = (40, 50, 70)
        self.COLOR_TILE_OUTLINE = (60, 70, 90)
        self.COLOR_PLAYER = (0, 150, 255)
        self.COLOR_PLAYER_GLOW = (0, 150, 255, 50)
        self.COLOR_CRYSTAL = (255, 255, 255)
        self.COLOR_TRAP = (255, 20, 50)
        self.COLOR_TRAP_GLOW = (255, 20, 50, 70)
        self.COLOR_TRAP_DEAD = (10, 10, 10)
        self.COLOR_TEXT = (220, 220, 240)
        self.COLOR_WIN = (50, 255, 50)
        self.COLOR_LOSE = (255, 50, 50)

        # --- State Variables ---
        self.player_pos = (0, 0)
        self.crystals = []
        self.traps = []
        self.particles = deque()
        self.steps = 0
        self.score = 0
        self.crystals_collected = 0
        self.game_over = False
        self.win_state = False
        self.last_dist_to_crystal = float('inf')
        self.last_dist_to_trap = float('inf')
        
        # Initialize state
        self.reset()
        
        # Validate implementation
        self.validate_implementation()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.steps = 0
        self.score = 0
        self.crystals_collected = 0
        self.game_over = False
        self.win_state = False
        self.particles.clear()

        # Place player in the center
        self.player_pos = (self.grid_size[0] // 2, self.grid_size[1] // 2)

        # Generate unique positions for crystals and traps
        possible_positions = [(x, y) for x in range(self.grid_size[0]) for y in range(self.grid_size[1])]
        possible_positions.remove(self.player_pos)

        # Ensure traps are not too close to the start
        safe_trap_positions = [
            p for p in possible_positions 
            if abs(p[0] - self.player_pos[0]) + abs(p[1] - self.player_pos[1]) > 3
        ]
        
        self.np_random.shuffle(possible_positions)
        self.np_random.shuffle(safe_trap_positions)

        self.traps = safe_trap_positions[:self.num_traps]
        
        # Place crystals in remaining empty spots
        crystal_positions = [p for p in possible_positions if p not in self.traps]
        self.crystals = crystal_positions[:self.num_crystals]

        # Pre-calculate initial distances for reward shaping
        self.last_dist_to_crystal = self._get_min_dist(self.player_pos, self.crystals)
        self.last_dist_to_trap = self._get_min_dist(self.player_pos, self.traps)

        return self._get_observation(), self._get_info()
    
    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        # space_held = action[1] == 1 # Unused
        # shift_held = action[2] == 1 # Unused
        
        # --- Update Game Logic ---
        self.steps += 1
        reward = 0
        
        # Move player
        old_pos = self.player_pos
        dx, dy = 0, 0
        if movement == 1: # Up (Up-Left)
            dx, dy = -1, 0
        elif movement == 2: # Down (Down-Right)
            dx, dy = 1, 0
        elif movement == 3: # Left (Down-Left)
            dx, dy = 0, -1
        elif movement == 4: # Right (Up-Right)
            dx, dy = 0, 1

        if movement != 0:
            new_pos_x = np.clip(self.player_pos[0] + dx, 0, self.grid_size[0] - 1)
            new_pos_y = np.clip(self.player_pos[1] + dy, 0, self.grid_size[1] - 1)
            self.player_pos = (new_pos_x, new_pos_y)

        # --- Reward Shaping ---
        dist_to_crystal = self._get_min_dist(self.player_pos, self.crystals)
        if dist_to_crystal < self.last_dist_to_crystal:
            reward += 1
        self.last_dist_to_crystal = dist_to_crystal
        
        dist_to_trap = self._get_min_dist(self.player_pos, self.traps)
        if dist_to_trap < self.last_dist_to_trap:
            reward -= 1
        self.last_dist_to_trap = dist_to_trap

        # --- Collision & Events ---
        if self.player_pos in self.crystals:
            # // SFX: Crystal collect sound
            self.crystals.remove(self.player_pos)
            self.crystals_collected += 1
            reward += 10
            self.score += 10
            self._create_particles(self.player_pos, self.COLOR_CRYSTAL, 15)
            # Recalculate dist to next crystal
            self.last_dist_to_crystal = self._get_min_dist(self.player_pos, self.crystals)

        terminated = False
        if self.player_pos in self.traps:
            # // SFX: Player falls into trap sound
            self.game_over = True
            self.win_state = False
            terminated = True
            reward = -100
            self.score -= 100

        if self.crystals_collected >= self.win_crystal_count:
            # // SFX: Victory fanfare
            self.game_over = True
            self.win_state = True
            terminated = True
            reward = 100
            self.score += 100

        if self.steps >= self.max_steps:
            terminated = True
            self.game_over = True # End game at max steps
        
        self.score += reward # Add shaping rewards to score for more granularity

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
            "crystals_collected": self.crystals_collected,
        }

    # --- Rendering Methods ---

    def _grid_to_iso(self, gx, gy):
        sx = self.origin_x + (gx - gy) * self.tile_width / 2
        sy = self.origin_y + (gx + gy) * self.tile_height / 2
        return int(sx), int(sy)

    def _draw_iso_tile(self, surface, pos, color, outline_color=None, offset_y=0):
        px, py = self._grid_to_iso(pos[0], pos[1])
        py += offset_y
        points = [
            (px, py - self.tile_height / 2),
            (px + self.tile_width / 2, py),
            (px, py + self.tile_height / 2),
            (px - self.tile_width / 2, py),
        ]
        pygame.gfxdraw.filled_polygon(surface, points, color)
        if outline_color:
            pygame.gfxdraw.aapolygon(surface, points, outline_color)

    def _update_and_draw_particles(self):
        for _ in range(len(self.particles)):
            particle = self.particles.popleft()
            particle['pos'][0] += particle['vel'][0]
            particle['pos'][1] += particle['vel'][1]
            particle['life'] -= 1
            if particle['life'] > 0:
                self.particles.append(particle)
                size = int(particle['life'] / particle['max_life'] * 4)
                if size > 0:
                    pygame.draw.circle(self.screen, particle['color'], particle['pos'], size)

    def _render_game(self):
        # Draw grid tiles
        for x in range(self.grid_size[0]):
            for y in range(self.grid_size[1]):
                self._draw_iso_tile(self.screen, (x, y), self.COLOR_TILE, self.COLOR_TILE_OUTLINE)

        # Draw traps
        # // SFX: Trap humming sound
        pulse = (math.sin(self.steps * 0.15) + 1) / 2  # 0 to 1
        trap_glow_radius = int(self.tile_width * 0.5 + pulse * 5)
        for trap_pos in self.traps:
            center_x, center_y = self._grid_to_iso(trap_pos[0], trap_pos[1])
            
            # Draw glow
            glow_surf = pygame.Surface((trap_glow_radius * 2, trap_glow_radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, self.COLOR_TRAP_GLOW, (trap_glow_radius, trap_glow_radius), trap_glow_radius)
            self.screen.blit(glow_surf, (center_x - trap_glow_radius, center_y - trap_glow_radius))
            
            if self.game_over and self.player_pos == trap_pos:
                self._draw_iso_tile(self.screen, trap_pos, self.COLOR_TRAP_DEAD)
            else:
                self._draw_iso_tile(self.screen, trap_pos, self.COLOR_TRAP)

        # Draw crystals
        sparkle_offset = (math.sin(self.steps * 0.3) * 3, math.cos(self.steps * 0.3) * 3)
        for crystal_pos in self.crystals:
            center_x, center_y = self._grid_to_iso(crystal_pos[0], crystal_pos[1])
            size = int(self.tile_width * 0.2)
            points = [
                (center_x, center_y - size),
                (center_x + size, center_y),
                (center_x, center_y + size),
                (center_x - size, center_y),
            ]
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_CRYSTAL)
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_CRYSTAL)
            # Sparkle effect
            pygame.draw.circle(self.screen, self.COLOR_CRYSTAL, (center_x + sparkle_offset[0], center_y - sparkle_offset[1]), 2)

        # Draw particles
        self._update_and_draw_particles()

        # Draw player
        player_bob = math.sin(self.steps * 0.2) * 3
        player_glow_radius = int(self.tile_width * 0.6)
        center_x, center_y = self._grid_to_iso(self.player_pos[0], self.player_pos[1])
        center_y += player_bob

        # Draw glow
        glow_surf = pygame.Surface((player_glow_radius * 2, player_glow_radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, self.COLOR_PLAYER_GLOW, (player_glow_radius, player_glow_radius), player_glow_radius)
        self.screen.blit(glow_surf, (center_x - player_glow_radius, center_y - player_glow_radius))
        
        self._draw_iso_tile(self.screen, self.player_pos, self.COLOR_PLAYER, offset_y=player_bob)

    def _render_ui(self):
        score_text = f"SCORE: {int(self.score)}"
        crystal_text = f"CRYSTALS: {self.crystals_collected} / {self.win_crystal_count}"
        
        score_surf = self.font_ui.render(score_text, True, self.COLOR_TEXT)
        crystal_surf = self.font_ui.render(crystal_text, True, self.COLOR_TEXT)

        self.screen.blit(score_surf, (10, 10))
        self.screen.blit(crystal_surf, (10, 40))

        if self.game_over:
            if self.win_state:
                msg, color = "YOU WIN!", self.COLOR_WIN
            else:
                msg, color = "GAME OVER", self.COLOR_LOSE
            
            end_surf = self.font_game_over.render(msg, True, color)
            end_rect = end_surf.get_rect(center=(self.screen_width / 2, self.screen_height / 2))
            self.screen.blit(end_surf, end_rect)

    # --- Helper Methods ---
    
    def _get_min_dist(self, pos, target_list):
        if not target_list:
            return float('inf')
        distances = [abs(pos[0] - t[0]) + abs(pos[1] - t[1]) for t in target_list]
        return min(distances)

    def _create_particles(self, grid_pos, color, count):
        px, py = self._grid_to_iso(grid_pos[0], grid_pos[1])
        for _ in range(count):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 3)
            life = self.np_random.integers(15, 30)
            self.particles.append({
                'pos': [px, py],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'color': color
            })

    def close(self):
        pygame.quit()

    def validate_implementation(self):
        '''
        Call this at the end of __init__ to verify implementation:
        '''
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
    # Set Pygame to use a visible display driver
    import os
    os.environ['SDL_VIDEODRIVER'] = 'x11' # Use 'windows' or 'x11' or 'dummy'

    env = GameEnv()
    obs, info = env.reset()
    
    # Create a visible window
    screen = pygame.display.set_mode((env.screen_width, env.screen_height))
    pygame.display.set_caption("Crystal Caverns")
    
    done = False
    
    # Game loop
    while not done:
        action = [0, 0, 0]  # Default to no-op
        
        # Pygame event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Keyboard input
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action[0] = 1
        elif keys[pygame.K_DOWN]:
            action[0] = 2
        elif keys[pygame.K_LEFT]:
            action[0] = 3
        elif keys[pygame.K_RIGHT]:
            action[0] = 4
        
        if keys[pygame.K_SPACE]:
            action[1] = 1
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            action[2] = 1
            
        # If any key is pressed, step the environment
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"Game Over! Final Score: {info['score']}, Steps: {info['steps']}")
                # Optional: pause for a moment then reset
                pygame.time.wait(2000)
                obs, info = env.reset()

        # Render the observation to the visible screen
        # Need to transpose back from (H, W, C) to (W, H, C) for pygame
        surf = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
        screen.blit(surf, (0, 0))
        pygame.display.flip()
        
        # Control the frame rate
        env.clock.tick(15) # Limit to 15 FPS for manual play

    env.close()