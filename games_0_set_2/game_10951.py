import gymnasium as gym
import os
import pygame
import os
import pygame

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")


# Generated: 2025-08-26T15:21:49.518405
# Source Brief: brief_00951.md
# Brief Index: 951
# """import gymnasium as gym
from gymnasium.spaces import MultiDiscrete
import numpy as np
import pygame
import pygame.gfxdraw
import math
import random

class GameEnv(gym.Env):
    """
    Gymnasium environment for a temporal puzzle game.

    The player controls a character (DK) and can place static temporal
    duplicates of themselves to block rolling barrels. The goal is to
    block all barrels in a level before they reach the bottom of the screen.
    """
    metadata = {"render_modes": ["rgb_array"]}
    
    game_description = (
        "A temporal puzzle game where you place static duplicates of yourself to block rolling barrels before they reach the bottom of the screen."
    )
    user_guide = (
        "Controls: Use arrow keys (↑↓←→) to move. Press space to place a temporal duplicate to block barrels."
    )
    auto_advance = True


    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # === Gymnasium Spaces ===
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(400, 640, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # === Pygame Setup ===
        self.WIDTH, self.HEIGHT = 640, 400
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_big = pygame.font.Font(None, 72)

        # === Visuals & Colors ===
        self.COLOR_BG = (10, 5, 10)
        self.COLOR_SCAFFOLD = (70, 40, 20)
        self.COLOR_PLAYER = (160, 82, 45)
        self.COLOR_PLAYER_TIE = (255, 0, 0)
        self.COLOR_DUPLICATE = (110, 62, 25, 180)
        self.COLOR_BARREL = (200, 20, 20)
        self.COLOR_BARREL_STRIPE = (220, 120, 120)
        self.COLOR_UI_TEXT = (255, 255, 255)
        self.COLOR_DANGER_ZONE = (50, 0, 0)
        self.COLOR_REWIND_PARTICLE = (100, 200, 255)

        # === Game Constants ===
        self.PLAYER_SIZE = 24
        self.PLAYER_SPEED = 24
        self.BARREL_SIZE = 16
        self.BARREL_SPEED = 2.0
        self.MAX_STEPS = 1000
        self.DANGER_ZONE_Y = self.HEIGHT - 20

        # === Game State (initialized in reset) ===
        self.level = 0
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level_won = False
        self.player_pos = [0, 0]
        self.duplicates = []
        self.active_barrels = []
        self.blocked_barrels = []
        self.total_level_barrels = 0
        self.rewinds_left = 0
        self.prev_space_held = False
        self.particles = []
        
        # Events for reward calculation
        self.newly_blocked_count = 0
        
        # Initialize state for the first time
        # self.reset() # reset is called by the wrapper, no need to call it here

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Level progression
        if not hasattr(self, 'level_won') or self.level_won:
            self.level += 1
        
        # Reset game state variables
        self.steps = 0
        self.score = 0
        self.game_over = False
        self.level_won = False
        self.prev_space_held = False
        
        # Generate level configuration
        num_barrels, self.rewinds_left, paths = self._generate_level_config(self.level)
        self.total_level_barrels = num_barrels
        
        # Initialize entities
        self.player_pos = [self.WIDTH // 2, self.HEIGHT - self.PLAYER_SIZE - 20]
        self.duplicates = []
        self.blocked_barrels = []
        self.particles = []

        # Create barrel objects
        self.active_barrels = []
        for i in range(num_barrels):
            path_index = i % len(paths)
            barrel = {
                "path": paths[path_index],
                "path_idx": 0,
                "pos": np.array(paths[path_index][0], dtype=float),
                "delay": i * 60 # Stagger barrel spawns
            }
            self.active_barrels.append(barrel)

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        self.steps += 1
        self.newly_blocked_count = 0

        # Handle actions
        self._handle_actions(action)

        # Update game state
        self._update_game_state()

        # Calculate reward and termination status
        reward = self._calculate_reward()
        self.score += reward
        terminated = self._check_termination()
        truncated = self.steps >= self.MAX_STEPS

        if truncated:
            self.game_over = True

        return (
            self._get_observation(),
            reward,
            terminated,
            truncated,
            self._get_info()
        )

    def _handle_actions(self, action):
        movement, space_held, _ = action
        
        # --- Movement ---
        px, py = self.player_pos
        if movement == 1: # Up
            py -= self.PLAYER_SPEED
        elif movement == 2: # Down
            py += self.PLAYER_SPEED
        elif movement == 3: # Left
            px -= self.PLAYER_SPEED
        elif movement == 4: # Right
            px += self.PLAYER_SPEED
        
        # Clamp player position to screen bounds
        self.player_pos[0] = max(0, min(self.WIDTH - self.PLAYER_SIZE, px))
        self.player_pos[1] = max(0, min(self.HEIGHT - self.PLAYER_SIZE, py))

        # --- Rewind (Space Press) ---
        is_rewind_press = space_held and not self.prev_space_held
        if is_rewind_press and self.rewinds_left > 0:
            self.rewinds_left -= 1
            # # SFX: Rewind_Sound()
            self.duplicates.append(self.player_pos.copy())
            self._create_rewind_particles(self.player_pos)
        self.prev_space_held = (space_held == 1)

    def _update_game_state(self):
        # --- Update Barrels ---
        barrels_to_remove = []
        for i, barrel in enumerate(self.active_barrels):
            if barrel["delay"] > 0:
                barrel["delay"] -= 1
                continue

            # Check for collision with player or duplicates
            is_blocked = False
            blockers = [self.player_pos] + self.duplicates
            for blocker_pos in blockers:
                if self._check_collision(barrel, blocker_pos):
                    is_blocked = True
                    break
            
            if is_blocked:
                # # SFX: Barrel_Block_Sound()
                barrels_to_remove.append(i)
                self.blocked_barrels.append(barrel)
                self.newly_blocked_count += 1
                continue

            # Move barrel along its path
            if barrel["path_idx"] < len(barrel["path"]) - 1:
                start_pos = np.array(barrel["path"][barrel["path_idx"]])
                end_pos = np.array(barrel["path"][barrel["path_idx"] + 1])
                direction = end_pos - start_pos
                dist = np.linalg.norm(direction)
                
                if dist > 0:
                    direction_norm = direction / dist
                    barrel["pos"] += direction_norm * self.BARREL_SPEED
                    
                    # Check if waypoint is reached
                    if np.linalg.norm(barrel["pos"] - start_pos) >= dist:
                        barrel["path_idx"] += 1
                        barrel["pos"] = end_pos
            
            # Check for failure condition
            if barrel["pos"][1] > self.DANGER_ZONE_Y:
                # # SFX: Level_Fail_Sound()
                self.game_over = True
                self.level_won = False

        # Remove blocked barrels from active list
        for i in sorted(barrels_to_remove, reverse=True):
            del self.active_barrels[i]

        # --- Update Particles ---
        self.particles = [p for p in self.particles if p['life'] > 0]
        for p in self.particles:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['life'] -= 1

    def _calculate_reward(self):
        reward = 0.0
        
        # Event-based reward for new blocks
        reward += self.newly_blocked_count * 1.0

        # Continuous reward for maintaining blocks
        reward += len(self.blocked_barrels) * 0.1
        
        # Check for terminal state
        if self._check_termination():
            if self.level_won:
                reward += 100.0
            else: # Loss or timeout
                reward -= 100.0
                
        return reward

    def _check_termination(self):
        # Win condition
        if len(self.blocked_barrels) == self.total_level_barrels:
            if not self.game_over: # First time win is detected
                # # SFX: Level_Win_Sound()
                self.level_won = True
                self.game_over = True
        
        # Lose condition (barrel reached bottom)
        # This is handled in _update_game_state, which sets self.game_over
            
        return self.game_over

    def _generate_level_config(self, level):
        num_barrels = 1 + (level - 1) // 2
        num_rewinds = max(1, 4 - (level - 1) // 3)
        
        # Path definitions
        path_straight = [ (self.WIDTH // 2, -20), (self.WIDTH // 2, self.HEIGHT + 20) ]
        path_zig_zag = [ (100, -20), (100, 150), (540, 150), (540, 300), (100, 300), (100, self.HEIGHT + 20) ]
        path_split_left = [ (self.WIDTH // 2, -20), (self.WIDTH // 2, 100), (150, 250), (150, self.HEIGHT + 20) ]
        path_split_right = [ (self.WIDTH // 2, -20), (self.WIDTH // 2, 100), (490, 250), (490, self.HEIGHT + 20) ]

        level_pattern = (level - 1) % 5
        if level_pattern == 0:
            paths = [path_straight]
        elif level_pattern == 1:
            paths = [path_zig_zag]
        elif level_pattern == 2:
            paths = [path_split_left, path_split_right]
        elif level_pattern == 3:
            paths = [path_zig_zag, path_straight]
        else:
            paths = [path_split_left, path_straight, path_split_right]

        return num_barrels, num_rewinds, paths

    def _check_collision(self, barrel, blocker_pos):
        barrel_center = barrel["pos"] + self.BARREL_SIZE / 2
        
        closest_x = max(blocker_pos[0], min(barrel_center[0], blocker_pos[0] + self.PLAYER_SIZE))
        closest_y = max(blocker_pos[1], min(barrel_center[1], blocker_pos[1] + self.PLAYER_SIZE))
        
        dist_x = barrel_center[0] - closest_x
        dist_y = barrel_center[1] - closest_y
        
        return (dist_x**2 + dist_y**2) < (self.BARREL_SIZE / 2)**2

    def _get_observation(self):
        self.screen.fill(self.COLOR_BG)
        self._render_background()
        self._render_game_elements()
        self._render_particles()
        self._render_ui()
        
        arr = pygame.surfarray.array3d(self.screen)
        return np.transpose(arr, (1, 0, 2)).astype(np.uint8)

    def _render_background(self):
        # Danger Zone
        pygame.draw.rect(self.screen, self.COLOR_DANGER_ZONE, (0, self.DANGER_ZONE_Y, self.WIDTH, self.HEIGHT - self.DANGER_ZONE_Y))
        # Scaffolding
        for y in range(50, self.HEIGHT, 80):
            pygame.draw.line(self.screen, self.COLOR_SCAFFOLD, (0, y), (self.WIDTH, y), 8)
        for x in range(80, self.WIDTH, 160):
            pygame.draw.line(self.screen, self.COLOR_SCAFFOLD, (x, 0), (x, self.HEIGHT), 8)

    def _render_game_elements(self):
        # Draw duplicates
        for pos in self.duplicates:
            self._draw_dk(pos, self.COLOR_DUPLICATE, is_duplicate=True)
            
        # Draw player
        self._draw_dk(self.player_pos, self.COLOR_PLAYER)

        # Draw barrels
        for barrel in self.active_barrels + self.blocked_barrels:
            if barrel["delay"] <= 0:
                self._draw_barrel(barrel["pos"])

    def _draw_dk(self, pos, color, is_duplicate=False):
        x, y = int(pos[0]), int(pos[1])
        s = self.PLAYER_SIZE
        if is_duplicate:
            surf = pygame.Surface((s, s), pygame.SRCALPHA)
            pygame.draw.rect(surf, color, (0, 0, s, s))
            pygame.draw.rect(surf, self.COLOR_PLAYER_TIE, (s//2 - 2, s//2, 4, 8))
            self.screen.blit(surf, (x, y))
        else:
            pygame.draw.rect(self.screen, color, (x, y, s, s))
            pygame.draw.rect(self.screen, self.COLOR_PLAYER_TIE, (x + s//2 - 2, y + s//2, 4, 8))

    def _draw_barrel(self, pos):
        x, y = int(pos[0]), int(pos[1])
        r = self.BARREL_SIZE // 2
        pygame.draw.circle(self.screen, self.COLOR_BARREL, (x + r, y + r), r)
        pygame.draw.circle(self.screen, (0,0,0), (x + r, y + r), r, 2)
        pygame.draw.rect(self.screen, self.COLOR_BARREL_STRIPE, (x, y + r - 2, self.BARREL_SIZE, 4))

    def _render_particles(self):
        for p in self.particles:
            alpha = max(0, min(255, int(255 * (p['life'] / p['max_life']))))
            color = (*self.COLOR_REWIND_PARTICLE, alpha)
            surf = pygame.Surface((p['size']*2, p['size']*2), pygame.SRCALPHA)
            pygame.gfxdraw.filled_circle(surf, p['size'], p['size'], p['size'], color)
            self.screen.blit(surf, (int(p['pos'][0]-p['size']), int(p['pos'][1]-p['size'])))

    def _create_rewind_particles(self, pos):
        center_x = pos[0] + self.PLAYER_SIZE / 2
        center_y = pos[1] + self.PLAYER_SIZE / 2
        for _ in range(30):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 4)
            life = random.randint(20, 40)
            self.particles.append({
                'pos': [center_x, center_y],
                'vel': [math.cos(angle) * speed, math.sin(angle) * speed],
                'life': life,
                'max_life': life,
                'size': random.randint(2, 5),
            })
            
    def _render_ui(self):
        # UI Panel background
        ui_panel = pygame.Surface((self.WIDTH, 30), pygame.SRCALPHA)
        ui_panel.fill((0, 0, 0, 150))
        self.screen.blit(ui_panel, (0, 0))

        # Rewinds
        rewind_text = self.font_ui.render(f"REWNDS: {self.rewinds_left}", True, self.COLOR_UI_TEXT)
        self.screen.blit(rewind_text, (10, 5))
        
        # Level
        level_text = self.font_ui.render(f"LEVEL: {self.level}", True, self.COLOR_UI_TEXT)
        self.screen.blit(level_text, (self.WIDTH // 2 - level_text.get_width() // 2, 5))
        
        # Score
        score_text = self.font_ui.render(f"SCORE: {int(self.score)}", True, self.COLOR_UI_TEXT)
        self.screen.blit(score_text, (self.WIDTH - score_text.get_width() - 10, 5))

        # Game Over / Win Message
        if self.game_over:
            overlay = pygame.Surface((self.WIDTH, self.HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            message = "LEVEL COMPLETE" if self.level_won else "GAME OVER"
            color = (100, 255, 100) if self.level_won else (255, 50, 50)
            
            end_text = self.font_big.render(message, True, color)
            text_rect = end_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/2))
            self.screen.blit(end_text, text_rect)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.steps,
            "level": self.level,
            "rewinds_left": self.rewinds_left,
            "barrels_blocked": len(self.blocked_barrels),
            "barrels_total": self.total_level_barrels,
        }
        
    def close(self):
        pygame.quit()

# The original code had a validation method that is not part of the standard
# gym.Env API and can cause issues with some environment wrappers.
# It's better to rely on external testing scripts for validation.
# If you need to run this file standalone, you can uncomment the following:
#
# if __name__ == '__main__':
#     # This block is for human play and visualization.
#     # It's not part of the Gymnasium environment definition.
#     # We need to unset the dummy video driver to see the window.
#     if "SDL_VIDEODRIVER" in os.environ:
#         del os.environ["SDL_VIDEODRIVER"]
#
#     env = GameEnv()
#     obs, info = env.reset()
#
#     running = True
#     terminated = False
#     truncated = False
#
#     # Use a display screen for human play
#     display_screen = pygame.display.set_mode((env.WIDTH, env.HEIGHT))
#     pygame.display.set_caption("Temporal DK")
#
#     action = env.action_space.sample() # Start with a no-op
#     action[0] = 0 # No movement
#     action[1] = 0 # Space released
#     action[2] = 0 # Shift released
#
#     while running:
#         for event in pygame.event.get():
#             if event.type == pygame.QUIT:
#                 running = False
#             if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
#                 print("Resetting environment")
#                 terminated = False
#                 truncated = False
#                 obs, info = env.reset()
#
#         if not (terminated or truncated):
#             keys = pygame.key.get_pressed()
#
#             # Map keys to actions
#             mov = 0 # None
#             if keys[pygame.K_UP] or keys[pygame.K_w]: mov = 1
#             elif keys[pygame.K_DOWN] or keys[pygame.K_s]: mov = 2
#             elif keys[pygame.K_LEFT] or keys[pygame.K_a]: mov = 3
#             elif keys[pygame.K_RIGHT] or keys[pygame.K_d]: mov = 4
#
#             space = 1 if keys[pygame.K_SPACE] else 0
#             shift = 1 if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT] else 0
#
#             action = [mov, space, shift]
#
#             obs, reward, terminated, truncated, info = env.step(action)
#
#             if reward != 0:
#                 print(f"Step: {info['steps']}, Reward: {reward:.2f}, Score: {info['score']:.2f}, Terminated: {terminated}, Truncated: {truncated}")
#
#             if terminated or truncated:
#                 print(f"Episode finished. Final Score: {info['score']}. Press 'R' to reset.")
#
#         # Render to the display screen
#         draw_surface = pygame.surfarray.make_surface(np.transpose(obs, (1, 0, 2)))
#         display_screen.blit(draw_surface, (0, 0))
#         pygame.display.flip()
#
#         env.clock.tick(30) # Limit FPS for human play
#
#     env.close()