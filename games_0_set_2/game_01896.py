
# Generated: 2025-08-27T18:38:02.798486
# Source Brief: brief_01896.md
# Brief Index: 1896

        
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

    # Must be a short, user-facing control string:
    user_guide = (
        "Controls: Use arrow keys to move. Collect all green gems while avoiding the red traps."
    )

    # Must be a short, user-facing description of the game:
    game_description = (
        "A strategic puzzle game. Navigate a grid to collect all the gems, but plan your moves carefully to avoid the hidden traps."
    )

    # Should frames auto-advance or wait for user input?
    auto_advance = False

    def __init__(self, render_mode="rgb_array"):
        super().__init__()

        # --- Game Constants ---
        self.SCREEN_WIDTH = 640
        self.SCREEN_HEIGHT = 400
        self.GRID_WIDTH = 16
        self.GRID_HEIGHT = 10
        self.CELL_SIZE = self.SCREEN_WIDTH // self.GRID_WIDTH
        
        self.NUM_GEMS = 20
        self.NUM_TRAPS = 15
        self.MAX_TRAPS_HIT = 5
        self.MAX_STEPS = 200

        # --- Colors ---
        self.COLOR_BG = (20, 25, 30)
        self.COLOR_GRID = (40, 45, 50)
        self.COLOR_PLAYER = (60, 180, 255)
        self.COLOR_PLAYER_OUTLINE = (255, 255, 255)
        self.COLOR_GEM = (0, 255, 128)
        self.COLOR_TRAP = (255, 50, 80)
        self.COLOR_TEXT = (240, 240, 240)
        self.COLOR_POPUP_GOOD = (100, 255, 150)
        self.COLOR_POPUP_BAD = (255, 100, 100)
        
        # --- Gymnasium Spaces ---
        self.observation_space = Box(
            low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8
        )
        self.action_space = MultiDiscrete([5, 2, 2])

        # --- Pygame Setup ---
        pygame.init()
        pygame.font.init()
        self.screen = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font_ui = pygame.font.Font(None, 28)
        self.font_popup = pygame.font.Font(None, 24)
        self.font_game_over = pygame.font.Font(None, 72)

        # --- State Variables ---
        self.player_pos = None
        self.gem_locations = None
        self.trap_locations = None
        self.steps = 0
        self.score = 0
        self.traps_triggered = 0
        self.gems_collected = 0
        self.game_over = False
        self.game_won = False
        
        # --- Visual Effects ---
        self.particles = []
        self.reward_popups = []
        self.trap_flash_timer = 0
        
        self.reset()
        
        # self.validate_implementation() # Optional: Call for self-check

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.steps = 0
        self.score = 0
        self.traps_triggered = 0
        self.gems_collected = 0
        self.game_over = False
        self.game_won = False
        self.particles.clear()
        self.reward_popups.clear()

        # --- Generate Game Board ---
        self.player_pos = [self.GRID_WIDTH // 2, self.GRID_HEIGHT // 2]
        
        all_cells = [(x, y) for x in range(self.GRID_WIDTH) for y in range(self.GRID_HEIGHT)]
        all_cells.remove(tuple(self.player_pos))
        
        # Traps cannot be within 3 Manhattan distance of the player's start
        safe_zone_radius = 3
        trap_valid_cells = [
            cell for cell in all_cells 
            if self._manhattan_distance(self.player_pos, cell) > safe_zone_radius
        ]
        self.np_random.shuffle(trap_valid_cells)
        self.trap_locations = set(trap_valid_cells[:self.NUM_TRAPS])

        # Gems can be anywhere except on traps or the player's start
        gem_valid_cells = [cell for cell in all_cells if cell not in self.trap_locations]
        self.np_random.shuffle(gem_valid_cells)
        self.gem_locations = set(gem_valid_cells[:self.NUM_GEMS])

        return self._get_observation(), self._get_info()

    def step(self, action):
        if self.game_over:
            return self._get_observation(), 0, True, False, self._get_info()

        movement = action[0]
        reward = 0

        # --- Calculate Proximity Reward ---
        old_pos = list(self.player_pos)
        dist_before_gem = self._get_nearest_entity_dist(old_pos, self.gem_locations)
        dist_before_trap = self._get_nearest_entity_dist(old_pos, self.trap_locations)

        # --- Apply Movement ---
        if movement == 1:  # Up
            self.player_pos[1] = max(0, self.player_pos[1] - 1)
        elif movement == 2:  # Down
            self.player_pos[1] = min(self.GRID_HEIGHT - 1, self.player_pos[1] + 1)
        elif movement == 3:  # Left
            self.player_pos[0] = max(0, self.player_pos[0] - 1)
        elif movement == 4:  # Right
            self.player_pos[0] = min(self.GRID_WIDTH - 1, self.player_pos[0] + 1)
        
        dist_after_gem = self._get_nearest_entity_dist(self.player_pos, self.gem_locations)
        dist_after_trap = self._get_nearest_entity_dist(self.player_pos, self.trap_locations)
        
        # Reward for getting closer to a gem
        if dist_after_gem < dist_before_gem:
            reward += 1
        # Penalty for getting closer to a trap
        if dist_after_trap < dist_before_trap:
            reward -= 1

        # --- Check for Events ---
        player_pos_tuple = tuple(self.player_pos)
        
        # Gem Collection
        if player_pos_tuple in self.gem_locations:
            self.gem_locations.remove(player_pos_tuple)
            gem_reward = 10
            reward += gem_reward
            self.score += gem_reward
            self.gems_collected += 1
            self._create_gem_particles(player_pos_tuple)
            self._create_reward_popup(f"+{gem_reward}", player_pos_tuple, self.COLOR_POPUP_GOOD)
            # sfx: gem collect sound

        # Trap Trigger
        if player_pos_tuple in self.trap_locations:
            trap_penalty = -20
            reward += trap_penalty
            self.score += trap_penalty
            self.traps_triggered += 1
            self.trap_flash_timer = 5 # Visual effect duration
            self._create_reward_popup(f"{trap_penalty}", player_pos_tuple, self.COLOR_POPUP_BAD)
            # sfx: trap trigger sound

        self.steps += 1
        
        # --- Check Termination Conditions ---
        terminated = False
        if self.gems_collected >= self.NUM_GEMS:
            win_reward = 100
            reward += win_reward
            self.score += win_reward
            terminated = True
            self.game_over = True
            self.game_won = True
            # sfx: victory fanfare

        if self.traps_triggered >= self.MAX_TRAPS_HIT:
            loss_penalty = -100
            reward += loss_penalty
            self.score += loss_penalty
            terminated = True
            self.game_over = True
            # sfx: game over sound

        if self.steps >= self.MAX_STEPS:
            terminated = True
            self.game_over = True

        return self._get_observation(), reward, terminated, False, self._get_info()

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
            "gems_collected": self.gems_collected,
            "traps_triggered": self.traps_triggered,
        }

    def _grid_to_pixel(self, grid_pos):
        px = grid_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
        py = grid_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        return int(px), int(py)

    def _manhattan_distance(self, p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _get_nearest_entity_dist(self, pos, entity_set):
        if not entity_set:
            return float('inf')
        return min(self._manhattan_distance(pos, entity) for entity in entity_set)

    def _render_game(self):
        # --- Draw Grid ---
        for x in range(self.GRID_WIDTH + 1):
            px = x * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (px, 0), (px, self.SCREEN_HEIGHT), 1)
        for y in range(self.GRID_HEIGHT + 1):
            py = y * self.CELL_SIZE
            pygame.draw.line(self.screen, self.COLOR_GRID, (0, py), (self.SCREEN_WIDTH, py), 1)

        # --- Draw Traps ---
        for trap_pos in self.trap_locations:
            px, py = self._grid_to_pixel(trap_pos)
            rect = pygame.Rect(px - self.CELL_SIZE // 4, py - self.CELL_SIZE // 4, self.CELL_SIZE // 2, self.CELL_SIZE // 2)
            pygame.draw.rect(self.screen, self.COLOR_TRAP, rect)

        # --- Draw Gems ---
        for gem_pos in self.gem_locations:
            px, py = self._grid_to_pixel(gem_pos)
            radius = self.CELL_SIZE // 3
            points = [
                (px, py - radius),
                (px + radius, py),
                (px, py + radius),
                (px - radius, py),
            ]
            pygame.gfxdraw.aapolygon(self.screen, points, self.COLOR_GEM)
            pygame.gfxdraw.filled_polygon(self.screen, points, self.COLOR_GEM)
            
        # --- Draw Player ---
        px, py = self._grid_to_pixel(self.player_pos)
        radius = self.CELL_SIZE // 3
        pygame.gfxdraw.aacircle(self.screen, px, py, radius, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius, self.COLOR_PLAYER_OUTLINE)
        pygame.gfxdraw.aacircle(self.screen, px, py, radius - 2, self.COLOR_PLAYER)
        pygame.gfxdraw.filled_circle(self.screen, px, py, radius - 2, self.COLOR_PLAYER)

        # --- Update and Draw Effects ---
        self._update_and_draw_particles()
        self._update_and_draw_popups()
        
        # --- Trap Flash Effect ---
        if self.trap_flash_timer > 0:
            flash_surface = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            alpha = int(100 * (self.trap_flash_timer / 5.0))
            flash_surface.fill((*self.COLOR_TRAP, alpha))
            self.screen.blit(flash_surface, (0, 0))
            self.trap_flash_timer -= 1

    def _render_ui(self):
        # Score
        score_text = self.font_ui.render(f"Score: {self.score}", True, self.COLOR_TEXT)
        self.screen.blit(score_text, (10, 10))

        # Traps Triggered
        trap_text = self.font_ui.render(f"Traps: {self.traps_triggered}/{self.MAX_TRAPS_HIT}", True, self.COLOR_TEXT)
        text_rect = trap_text.get_rect(topright=(self.SCREEN_WIDTH - 10, 10))
        self.screen.blit(trap_text, text_rect)
        
        # Game Over Message
        if self.game_over:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.screen.blit(overlay, (0, 0))
            
            if self.game_won:
                end_text = self.font_game_over.render("YOU WIN!", True, self.COLOR_GEM)
            else:
                end_text = self.font_game_over.render("GAME OVER", True, self.COLOR_TRAP)
                
            text_rect = end_text.get_rect(center=(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2))
            self.screen.blit(end_text, text_rect)

    def _create_gem_particles(self, grid_pos):
        px, py = self._grid_to_pixel(grid_pos)
        for _ in range(15):
            angle = self.np_random.uniform(0, 2 * math.pi)
            speed = self.np_random.uniform(1, 4)
            vel = [math.cos(angle) * speed, math.sin(angle) * speed]
            lifetime = self.np_random.integers(20, 40)
            self.particles.append({'pos': [px, py], 'vel': vel, 'lifetime': lifetime, 'max_life': lifetime, 'color': self.COLOR_GEM})

    def _update_and_draw_particles(self):
        for p in self.particles[:]:
            p['pos'][0] += p['vel'][0]
            p['pos'][1] += p['vel'][1]
            p['vel'][1] += 0.1 # Gravity
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.particles.remove(p)
            else:
                alpha = int(255 * (p['lifetime'] / p['max_life']))
                color = (*p['color'], alpha)
                size = max(1, int(5 * (p['lifetime'] / p['max_life'])))
                rect = pygame.Rect(int(p['pos'][0]), int(p['pos'][1]), size, size)
                
                # Create a temporary surface to draw the particle with alpha
                temp_surf = pygame.Surface((size, size), pygame.SRCALPHA)
                temp_surf.fill(color)
                self.screen.blit(temp_surf, rect.topleft)

    def _create_reward_popup(self, text, grid_pos, color):
        px, py = self._grid_to_pixel(grid_pos)
        self.reward_popups.append({'text': text, 'pos': [px, py], 'lifetime': 60, 'max_life': 60, 'color': color})

    def _update_and_draw_popups(self):
        for p in self.reward_popups[:]:
            p['pos'][1] -= 0.5 # Move up
            p['lifetime'] -= 1
            if p['lifetime'] <= 0:
                self.reward_popups.remove(p)
            else:
                alpha = int(255 * (p['lifetime'] / p['max_life']))
                color = (*p['color'][:3], alpha)
                text_surf = self.font_popup.render(p['text'], True, color)
                text_surf.set_alpha(alpha)
                text_rect = text_surf.get_rect(center=(int(p['pos'][0]), int(p['pos'][1])))
                self.screen.blit(text_surf, text_rect)

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
    env = GameEnv(render_mode="rgb_array")
    obs, info = env.reset()
    
    # Setup Pygame window for human play
    pygame.display.set_caption("Gem Collector")
    screen = pygame.display.set_mode((env.SCREEN_WIDTH, env.SCREEN_HEIGHT))
    clock = pygame.time.Clock()
    
    done = False
    print(env.user_guide)
    
    while not done:
        action = [0, 0, 0] # Default action: no-op
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
        
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

        # In a turn-based game, we only step if a move key is pressed
        if action[0] != 0:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            print(f"Action: {action}, Reward: {reward:.2f}, Score: {info['score']}, Done: {done}")

        # --- Rendering ---
        # The environment's observation is already a rendered frame
        # We just need to display it.
        frame = np.transpose(obs, (1, 0, 2)) # Transpose back for pygame
        surf = pygame.surfarray.make_surface(frame)
        screen.blit(surf, (0, 0))
        
        pygame.display.flip()
        clock.tick(10) # Limit frame rate for human play

    env.close()
    pygame.quit()